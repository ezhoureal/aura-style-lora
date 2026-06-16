from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image
from peft import LoraConfig
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

from lora.local_edit_common import (
    REPO_ROOT,
    SUPPORTED_IMAGE_SUFFIXES,
    apply_lora_checkpoint,
    checkpoint_dir_for_step,
    configure_environment,
    configure_training_runtime,
    dtype_from_precision,
    freeze_module,
    load_rgb_image,
    make_accelerator,
    make_lr_scheduler,
    make_optimizer,
    make_pair_image_transform,
    make_training_plan,
    none_if_null,
    read_metadata,
    resolve_repo_path,
    resolve_resume_checkpoint,
    run_training_loop,
    trainable_parameters,
)


@dataclass(frozen=True)
class StyleExample:
    image_path: Path
    prompt: str


@dataclass(frozen=True)
class StyleBatch:
    pixel_values: Tensor
    input_ids: Tensor


def fallback_prompt(cfg: DictConfig) -> str:
    return str(cfg.prompt.fallback_template).format(
        trigger=str(cfg.prompt.trigger),
        style_description=str(cfg.prompt.style_description),
    )


def style_prompt(cfg: DictConfig, row_prompt: str) -> str:
    trigger = str(cfg.prompt.trigger)
    if trigger in row_prompt:
        return row_prompt
    return row_prompt.replace("aura style", f"{trigger} style")


def load_style_examples(cfg: DictConfig) -> list[StyleExample]:
    train_dir = resolve_repo_path(str(cfg.dataset.train_dir))
    metadata_path = train_dir / "metadata.jsonl"
    rows = read_metadata(metadata_path)
    examples: list[StyleExample] = []
    default_prompt = fallback_prompt(cfg)

    for row in rows:
        if row.get(str(cfg.dataset.kind_key)) != str(cfg.dataset.paired_kind):
            continue
        image_name = row.get(str(cfg.dataset.image_key))
        if not isinstance(image_name, str):
            continue
        image_path = train_dir / image_name
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if not image_path.exists():
            raise FileNotFoundError(f"Missing style training image: {image_path}")
        configured_prompt = row.get(str(cfg.dataset.prompt_key))
        prompt = (
            style_prompt(cfg, configured_prompt)
            if isinstance(configured_prompt, str)
            else default_prompt
        )
        examples.append(StyleExample(image_path=image_path, prompt=prompt))

    if not examples:
        raise ValueError(f"No style training examples found in {metadata_path}")
    return examples


class SD15StyleDataset(Dataset[StyleBatch]):
    def __init__(
        self,
        examples: list[StyleExample],
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

    def __getitem__(self, index: int) -> StyleBatch:
        example = self.examples[index]
        image = load_rgb_image(example.image_path)
        if self.random_flip and torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tokens = self.tokenizer(
            example.prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return StyleBatch(
            pixel_values=cast(Tensor, self.image_transform(image)),
            input_ids=cast(Tensor, tokens.input_ids[0]),
        )


def collate_style_batches(items: list[StyleBatch]) -> StyleBatch:
    return StyleBatch(
        pixel_values=torch.stack([item.pixel_values for item in items]),
        input_ids=torch.stack([item.input_ids for item in items]),
    )


def model_load_kwargs(model_cfg: DictConfig, local_files_only: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    revision = none_if_null(model_cfg.get("revision"))
    variant = none_if_null(model_cfg.get("variant"))
    if revision is not None:
        kwargs["revision"] = revision
    if variant is not None:
        kwargs["variant"] = variant
    return kwargs


def load_training_vae(cfg: DictConfig, weight_dtype: torch.dtype) -> AutoencoderKL:
    if bool(cfg.models.vae.enabled):
        return AutoencoderKL.from_pretrained(
            str(resolve_repo_path(str(cfg.models.vae.pretrained_model_name_or_path))),
            torch_dtype=weight_dtype,
            local_files_only=bool(cfg.models.vae.local_files_only),
        )
    return AutoencoderKL.from_pretrained(
        str(resolve_repo_path(str(cfg.models.base.pretrained_model_name_or_path))),
        subfolder="vae",
        torch_dtype=weight_dtype,
        local_files_only=bool(cfg.training.local_files_only),
    )


def upcast_trainable_parameters(module: UNet2DConditionModel) -> None:
    for parameter in trainable_parameters(module):
        parameter.data = parameter.data.to(torch.float32)


def write_pipeline_lora_weights(save_dir: Path) -> Path:
    raw_path = save_dir / "pytorch_lora_weights.safetensors"
    raw_state = load_file(raw_path)
    pipeline_state = {f"unet.{key}": tensor for key, tensor in raw_state.items() if ".lora_" in key}
    if not pipeline_state:
        raise ValueError(f"No LoRA tensors found in {raw_path}")
    pipeline_path = save_dir / "pipeline_lora_weights.safetensors"
    save_file(
        pipeline_state,
        pipeline_path,
        metadata={"format": "pt", "source": "sd15_style_lora"},
    )
    return pipeline_path


class SD15StyleLoraTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.output_dir = resolve_repo_path(str(cfg.training.output_root))

    def train(self) -> Path:
        training_cfg = self.cfg.training
        model_path = resolve_repo_path(str(self.cfg.models.base.pretrained_model_name_or_path))
        weight_dtype = dtype_from_precision(str(training_cfg.mixed_precision))
        accelerator = make_accelerator(training_cfg, self.output_dir)
        configure_training_runtime(training_cfg)
        load_kwargs = model_load_kwargs(self.cfg.models.base, bool(training_cfg.local_files_only))

        tokenizer = CLIPTokenizer.from_pretrained(
            str(model_path), subfolder="tokenizer", **load_kwargs
        )
        noise_scheduler = DDPMScheduler.from_pretrained(
            str(model_path),
            subfolder="scheduler",
            **load_kwargs,
        )
        prediction_type = none_if_null(training_cfg.prediction_type)
        if prediction_type is not None:
            noise_scheduler.register_to_config(prediction_type=prediction_type)
        text_encoder = CLIPTextModel.from_pretrained(
            str(model_path),
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            **load_kwargs,
        )
        vae = load_training_vae(self.cfg, weight_dtype)
        unet = UNet2DConditionModel.from_pretrained(
            str(model_path),
            subfolder="unet",
            torch_dtype=weight_dtype,
            **load_kwargs,
        )

        freeze_module(vae)
        freeze_module(text_encoder)
        freeze_module(unet)
        unet.add_adapter(
            LoraConfig(
                r=int(training_cfg.rank),
                lora_alpha=int(training_cfg.lora_alpha),
                init_lora_weights="gaussian",
                target_modules=[str(item) for item in training_cfg.target_modules],
            )
        )
        if bool(training_cfg.gradient_checkpointing):
            unet.enable_gradient_checkpointing()
        if str(training_cfg.mixed_precision) == "fp16":
            upcast_trainable_parameters(unet)

        resume_checkpoint = resolve_resume_checkpoint(training_cfg, self.output_dir)
        if resume_checkpoint is not None:
            apply_lora_checkpoint(unet, resume_checkpoint.path)

        examples = load_style_examples(self.cfg)
        dataset = SD15StyleDataset(
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
            collate_fn=collate_style_batches,
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

        def loss_step(batch: StyleBatch) -> Tensor:
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
            model_key="sd15_style_lora",
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
        batch: StyleBatch,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        unet: UNet2DConditionModel,
        noise_scheduler: DDPMScheduler,
        weight_dtype: torch.dtype,
    ) -> Tensor:
        pixel_values = batch.pixel_values.to(accelerator.device, dtype=weight_dtype)
        input_ids = batch.input_ids.to(accelerator.device)
        encoded = cast(Any, vae.encode(pixel_values))
        latents = cast(Tensor, encoded.latent_dist.sample())
        latents = latents * float(cast(Any, vae.config).scaling_factor)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            int(cast(Any, noise_scheduler.config).num_train_timesteps),
            (latents.shape[0],),
            device=latents.device,
        )
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, cast(Any, timesteps))
        encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        target = noise
        if cast(Any, noise_scheduler.config).prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, cast(Any, timesteps))
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
        write_pipeline_lora_weights(save_dir)
        manifest = {
            "trainer": "sd15_style_lora",
            "step": step,
            "requires_source_image": False,
            "conditioning": "text_to_image_style_lora",
            "lora_rank": int(self.cfg.training.rank),
            "lora_alpha": int(self.cfg.training.lora_alpha),
            "trigger": str(self.cfg.prompt.trigger),
            "dataset_train_dir": str(self.cfg.dataset.train_dir),
            "base_model": str(self.cfg.models.base.pretrained_model_name_or_path),
        }
        with (save_dir / "training_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return save_dir


def run(cfg: DictConfig) -> None:
    configure_environment(cfg)
    output_dir = SD15StyleLoraTrainer(cfg).train()
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


def main() -> None:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="sd15_style_lora", overrides=sys.argv[1:])
    run(cfg)


__all__ = [
    "SD15StyleDataset",
    "SD15StyleLoraTrainer",
    "StyleBatch",
    "StyleExample",
    "collate_style_batches",
    "fallback_prompt",
    "load_style_examples",
    "main",
    "run",
    "style_prompt",
    "write_pipeline_lora_weights",
]
