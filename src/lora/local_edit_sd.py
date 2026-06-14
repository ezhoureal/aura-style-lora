from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import DictConfig
from PIL.Image import Image as PILImage
from peft import LoraConfig
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

from lora.local_edit_common import (
    PairExample,
    apply_lora_checkpoint,
    checkpoint_dir_for_step,
    configure_training_runtime,
    dtype_from_precision,
    evaluation_seed,
    freeze_module,
    generators_for_batch,
    load_condition_image,
    load_pair_examples,
    load_transformed_pair,
    make_accelerator,
    make_pair_image_transform,
    make_lr_scheduler,
    make_optimizer,
    make_training_plan,
    none_if_null,
    resolve_resume_checkpoint,
    run_training_loop,
    trainer_output_dir,
    write_training_manifest,
)


@dataclass(frozen=True)
class PreparedBatch:
    source_pixels: Tensor
    target_pixels: Tensor
    input_ids: Tensor


class PairedEditDataset(Dataset[PreparedBatch]):
    def __init__(
        self,
        examples: list[PairExample],
        tokenizer: CLIPTokenizer,
        resolution: int,
        center_crop: bool,
        random_flip: bool,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.image_transform = make_pair_image_transform(resolution, center_crop)
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreparedBatch:
        example = self.examples[index]
        source_pixels, target_pixels = load_transformed_pair(
            example,
            self.image_transform,
            self.random_flip,
        )

        tokens = self.tokenizer(
            example.prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return PreparedBatch(
            source_pixels=source_pixels,
            target_pixels=target_pixels,
            input_ids=cast(Tensor, tokens.input_ids[0]),
        )


def collate_batches(items: list[PreparedBatch]) -> PreparedBatch:
    return PreparedBatch(
        source_pixels=torch.stack([item.source_pixels for item in items]),
        target_pixels=torch.stack([item.target_pixels for item in items]),
        input_ids=torch.stack([item.input_ids for item in items]),
    )


def expand_unet_conv_in_for_ip2p(unet: UNet2DConditionModel) -> None:
    conv_in = unet.conv_in
    if conv_in.in_channels == 8:
        return
    if conv_in.in_channels != 4:
        raise ValueError(
            f"Expected UNet conv_in to have 4 or 8 channels, got {conv_in.in_channels}"
        )

    expanded = nn.Conv2d(
        8,
        conv_in.out_channels,
        kernel_size=cast(tuple[int, int], conv_in.kernel_size),
        stride=cast(tuple[int, int], conv_in.stride),
        padding=cast(tuple[int, int] | str, conv_in.padding),
    )
    expanded.to(device=conv_in.weight.device, dtype=conv_in.weight.dtype)
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :4].copy_(conv_in.weight)
        if conv_in.bias is not None and expanded.bias is not None:
            expanded.bias.copy_(conv_in.bias)
    unet.conv_in = expanded
    cast(Any, unet.config).in_channels = 8


def enable_unet_conv_in_training(unet: UNet2DConditionModel) -> None:
    unet.conv_in.requires_grad_(True)


class StableDiffusionIp2PLoraTrainer:
    def __init__(self, cfg: DictConfig, model_key: str) -> None:
        self.cfg = cfg
        self.model_key = model_key
        self.model_cfg = cfg.models[model_key]
        self.output_dir = trainer_output_dir(cfg, model_key)

    def train(self) -> Path:
        training_cfg = self.cfg.training
        model_id = str(self.model_cfg.pretrained_model_name_or_path)
        revision = none_if_null(self.model_cfg.get("revision"))
        variant = none_if_null(self.model_cfg.get("variant"))
        weight_dtype = dtype_from_precision(str(training_cfg.mixed_precision))
        accelerator = make_accelerator(training_cfg, self.output_dir)
        configure_training_runtime(training_cfg)

        load_kwargs: dict[str, Any] = {}
        if revision is not None:
            load_kwargs["revision"] = revision
        if variant is not None:
            load_kwargs["variant"] = variant
        load_kwargs["local_files_only"] = bool(training_cfg.local_files_only)
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            **load_kwargs,
        )
        noise_scheduler = DDPMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            **load_kwargs,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            **load_kwargs,
        )
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=weight_dtype,
            **load_kwargs,
        )
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=weight_dtype,
            **load_kwargs,
        )
        expand_unet_conv_in_for_ip2p(unet)
        freeze_module(vae)
        freeze_module(text_encoder)
        freeze_module(unet)
        unet.add_adapter(
            LoraConfig(
                r=int(training_cfg.rank),
                lora_alpha=int(training_cfg.lora_alpha),
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
        )
        enable_unet_conv_in_training(unet)
        if bool(training_cfg.gradient_checkpointing):
            unet.enable_gradient_checkpointing()

        resume_checkpoint = resolve_resume_checkpoint(training_cfg, self.output_dir)
        if resume_checkpoint is not None:
            apply_lora_checkpoint(unet, resume_checkpoint.path)
            apply_conv_in_patch_to_unet(unet, resume_checkpoint.path)

        examples = load_pair_examples(self.cfg.dataset)
        dataset = PairedEditDataset(
            examples=examples,
            tokenizer=tokenizer,
            resolution=int(self.cfg.dataset.resolution),
            center_crop=bool(self.cfg.dataset.center_crop),
            random_flip=bool(self.cfg.dataset.random_flip),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=int(training_cfg.train_batch_size),
            shuffle=True,
            collate_fn=collate_batches,
            num_workers=int(training_cfg.dataloader_num_workers),
        )
        optimizer = make_optimizer(unet, training_cfg)
        training_plan = make_training_plan(len(dataloader), training_cfg, resume_checkpoint)
        if training_plan.remaining_steps <= 0:
            final_dir = self.save(accelerator, unet, training_plan.initial_step)
            accelerator.end_training()
            return final_dir
        lr_scheduler = make_lr_scheduler(training_cfg, optimizer)

        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet,
            optimizer,
            dataloader,
            lr_scheduler,
        )
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        unet.train()
        unet_model = cast(UNet2DConditionModel, unet)

        def loss_step(batch: PreparedBatch) -> Tensor:
            return self.training_step(
                accelerator=accelerator,
                batch=batch,
                vae=vae,
                text_encoder=text_encoder,
                unet=unet_model,
                noise_scheduler=noise_scheduler,
                weight_dtype=weight_dtype,
            )

        def write_checkpoint(step: int) -> Path:
            return self.save(accelerator, unet_model, step)

        global_step = run_training_loop(
            accelerator=accelerator,
            train_model=unet_model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            training_cfg=training_cfg,
            model_key=self.model_key,
            plan=training_plan,
            loss_step=loss_step,
            write_checkpoint=write_checkpoint,
        )

        final_dir = self.save(accelerator, unet_model, global_step)
        accelerator.end_training()
        return final_dir

    def training_step(
        self,
        accelerator: Accelerator,
        batch: PreparedBatch,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        unet: UNet2DConditionModel,
        noise_scheduler: DDPMScheduler,
        weight_dtype: torch.dtype,
    ) -> Tensor:
        target_pixels = batch.target_pixels.to(accelerator.device, dtype=weight_dtype)
        source_pixels = batch.source_pixels.to(accelerator.device, dtype=weight_dtype)
        input_ids = batch.input_ids.to(accelerator.device)

        target_encoded = cast(Any, vae.encode(target_pixels))
        target_latents = cast(Tensor, target_encoded.latent_dist.sample())
        target_latents = target_latents * float(cast(Any, vae.config).scaling_factor)
        source_encoded = cast(Any, vae.encode(source_pixels))
        source_latents = cast(Tensor, source_encoded.latent_dist.mode())
        source_latents = source_latents * float(cast(Any, vae.config).scaling_factor)
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0,
            int(cast(Any, noise_scheduler.config).num_train_timesteps),
            (target_latents.shape[0],),
            device=target_latents.device,
        )
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(target_latents, noise, cast(Any, timesteps))
        encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]
        model_input = torch.cat([noisy_latents, source_latents], dim=1)
        model_pred = unet(model_input, timesteps, encoder_hidden_states, return_dict=False)[0]
        target = noise
        if cast(Any, noise_scheduler.config).prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(target_latents, noise, cast(Any, timesteps))
        return cast(Tensor, F.mse_loss(model_pred.float(), target.float(), reduction="mean"))

    def save(self, accelerator: Accelerator, unet: UNet2DConditionModel, step: int) -> Path:
        if not accelerator.is_main_process:
            return self.output_dir
        save_dir = checkpoint_dir_for_step(self.output_dir, step)
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        cast(UNet2DConditionModel, unwrapped_unet).save_lora_adapter(
            str(save_dir),
            adapter_name="default",
            safe_serialization=True,
        )
        conv_in = cast(UNet2DConditionModel, unwrapped_unet).conv_in
        save_file(
            {
                "weight": conv_in.weight.detach().cpu(),
                "bias": conv_in.bias.detach().cpu() if conv_in.bias is not None else torch.empty(0),
            },
            save_dir / "unet_conv_in_8ch.safetensors",
            metadata={"format": "pt", "source": "local_edit_lora"},
        )
        write_training_manifest(save_dir, self.cfg, self.model_key, self.model_cfg, step)
        return save_dir


def apply_conv_in_patch_to_unet(unet: UNet2DConditionModel, checkpoint_dir: Path) -> None:
    patch_path = checkpoint_dir / "unet_conv_in_8ch.safetensors"
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing image-edit conv_in patch: {patch_path}")
    state = load_file(patch_path)
    with torch.no_grad():
        unet.conv_in.weight.copy_(state["weight"].to(unet.conv_in.weight.device))
        bias = state["bias"]
        if unet.conv_in.bias is not None and bias.numel() > 0:
            unet.conv_in.bias.copy_(bias.to(unet.conv_in.bias.device))


def apply_conv_in_patch(pipe: StableDiffusionInstructPix2PixPipeline, checkpoint_dir: Path) -> None:
    apply_conv_in_patch_to_unet(pipe.unet, checkpoint_dir)


def load_sd_pipeline(
    cfg: DictConfig,
    model_key: str,
    checkpoint_dir: Path,
) -> StableDiffusionInstructPix2PixPipeline:
    model_cfg = cfg.models[model_key]
    dtype = dtype_from_precision(str(cfg.evaluation.mixed_precision))
    model_id = str(model_cfg.pretrained_model_name_or_path)
    load_kwargs: dict[str, Any] = {
        "local_files_only": bool(cfg.evaluation.local_files_only),
        "torch_dtype": dtype,
    }
    revision = none_if_null(model_cfg.get("revision"))
    variant = none_if_null(model_cfg.get("variant"))
    if revision is not None:
        load_kwargs["revision"] = revision
    if variant is not None:
        load_kwargs["variant"] = variant
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **load_kwargs)
    expand_unet_conv_in_for_ip2p(unet)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        unet=unet,
        safety_checker=None,
        **load_kwargs,
    )
    apply_conv_in_patch(pipe, checkpoint_dir)
    pipe.unet.load_lora_adapter(
        str(checkpoint_dir),
        prefix=None,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="aura",
    )
    pipe.unet.set_adapters(["aura"], weights=[float(cfg.evaluation.lora_scale)])
    return pipe


def run_sd_batch(
    pipe: StableDiffusionInstructPix2PixPipeline,
    cfg: DictConfig,
    input_paths: list[Path],
    device_name: str,
    batch_offset: int,
) -> list[PILImage]:
    width = int(cfg.evaluation.width)
    height = int(cfg.evaluation.height)
    prompts = [str(cfg.evaluation.prompt)] * len(input_paths)
    images = [load_condition_image(input_path, width, height) for input_path in input_paths]
    result = cast(
        Any,
        pipe(
            prompt=prompts,
            image=images,
            num_inference_steps=int(cfg.evaluation.num_inference_steps),
            guidance_scale=float(cfg.evaluation.guidance_scale),
            image_guidance_scale=float(cfg.evaluation.image_guidance_scale),
            generator=generators_for_batch(
                evaluation_seed(cfg),
                device_name,
                batch_offset,
                len(input_paths),
            ),
        ),
    )
    return cast(list[PILImage], result.images)
