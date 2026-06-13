from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    calculate_shift,
    retrieve_timesteps,
)
from omegaconf import DictConfig
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
from peft import LoraConfig
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lora.local_edit_common import (
    PairedPromptBatch,
    PairedPromptDataset,
    apply_lora_checkpoint,
    collate_paired_prompt_batches,
    dtype_from_precision,
    evaluation_seed,
    freeze_module,
    load_pair_examples,
    load_rgb_image,
    make_training_progress,
    none_if_null,
    resolve_resume_checkpoint,
    resolve_repo_path,
    trainable_parameters,
)


SD3PreparedBatch = PairedPromptBatch
SD3PairedEditDataset = PairedPromptDataset
collate_sd3_batches = collate_paired_prompt_batches


@dataclass(frozen=True)
class SD3PromptEmbeds:
    prompt_embeds: Tensor
    pooled_prompt_embeds: Tensor


def expand_sd3_transformer_input_for_paired_edit(transformer: nn.Module) -> None:
    pos_embed = cast(Any, transformer).pos_embed
    projection = cast(nn.Conv2d, pos_embed.proj)
    if projection.in_channels == 32:
        return
    if projection.in_channels != 16:
        raise ValueError(
            f"Expected SD3 transformer input projection to have 16 or 32 channels, "
            f"got {projection.in_channels}"
        )

    expanded = nn.Conv2d(
        32,
        projection.out_channels,
        kernel_size=cast(tuple[int, int], projection.kernel_size),
        stride=cast(tuple[int, int], projection.stride),
        padding=cast(tuple[int, int] | str, projection.padding),
        bias=projection.bias is not None,
    )
    expanded.to(device=projection.weight.device, dtype=projection.weight.dtype)
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :16].copy_(projection.weight)
        if projection.bias is not None and expanded.bias is not None:
            expanded.bias.copy_(projection.bias)
    pos_embed.proj = expanded
    cast(Any, transformer.config).in_channels = 32


def apply_sd3_input_projection_patch(transformer: nn.Module, checkpoint_dir: Path) -> None:
    patch_path = checkpoint_dir / "sd3_input_projection_32ch.safetensors"
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing SD3 paired-edit input projection patch: {patch_path}")
    state = load_file(patch_path)
    projection = cast(nn.Conv2d, cast(Any, transformer).pos_embed.proj)
    with torch.no_grad():
        projection.weight.copy_(state["weight"].to(projection.weight.device))
        bias = state["bias"]
        if projection.bias is not None and bias.numel() > 0:
            projection.bias.copy_(bias.to(projection.bias.device))


class StableDiffusion3PairedEditLoraTrainer:
    def __init__(self, cfg: DictConfig, model_key: str) -> None:
        self.cfg = cfg
        self.model_key = model_key
        self.model_cfg = cfg.models[model_key]
        output_root = resolve_repo_path(str(cfg.training.output_root))
        self.output_dir = output_root / model_key

    def train(self) -> Path:
        training_cfg = self.cfg.training
        model_id = str(self.model_cfg.pretrained_model_name_or_path)
        weight_dtype = dtype_from_precision(str(training_cfg.mixed_precision))
        logging_dir = self.output_dir / "logs"
        accelerator = Accelerator(
            gradient_accumulation_steps=int(training_cfg.gradient_accumulation_steps),
            mixed_precision=str(training_cfg.mixed_precision),
            project_config=ProjectConfiguration(
                project_dir=str(self.output_dir),
                logging_dir=str(logging_dir),
            ),
        )
        set_seed(int(training_cfg.seed))
        if bool(training_cfg.allow_tf32):
            torch.backends.cuda.matmul.allow_tf32 = True

        load_kwargs: dict[str, Any] = {
            "torch_dtype": weight_dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": bool(training_cfg.get("local_files_only", False)),
        }
        if bool(training_cfg.get("sd3_disable_t5", False)):
            load_kwargs["text_encoder_3"] = None
            load_kwargs["tokenizer_3"] = None
        revision = none_if_null(self.model_cfg.get("revision"))
        variant = none_if_null(self.model_cfg.get("variant"))
        if revision is not None:
            load_kwargs["revision"] = revision
        if variant is not None:
            load_kwargs["variant"] = variant

        pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
        freeze_module(pipe.vae)
        freeze_module(pipe.text_encoder)
        freeze_module(pipe.text_encoder_2)
        if pipe.text_encoder_3 is not None:
            freeze_module(pipe.text_encoder_3)
        freeze_module(pipe.transformer)
        pipe.vae.eval()
        pipe.text_encoder.eval()
        pipe.text_encoder_2.eval()
        if pipe.text_encoder_3 is not None:
            pipe.text_encoder_3.eval()

        expand_sd3_transformer_input_for_paired_edit(cast(nn.Module, pipe.transformer))
        pipe.transformer.train()
        cast(Any, pipe.transformer).add_adapter(
            LoraConfig(
                r=int(training_cfg.rank),
                lora_alpha=int(training_cfg.lora_alpha),
                init_lora_weights="gaussian",
                target_modules=list(training_cfg.sd3_target_modules),
            )
        )
        if bool(training_cfg.gradient_checkpointing):
            pipe.transformer.enable_gradient_checkpointing()

        resume_checkpoint = resolve_resume_checkpoint(training_cfg, self.output_dir)
        if resume_checkpoint is not None:
            apply_lora_checkpoint(cast(nn.Module, pipe.transformer), resume_checkpoint.path)
            apply_sd3_input_projection_patch(
                cast(nn.Module, pipe.transformer), resume_checkpoint.path
            )

        examples = load_pair_examples(self.cfg.dataset)
        dataset = SD3PairedEditDataset(
            examples=examples,
            resolution=int(self.cfg.dataset.resolution),
            center_crop=bool(self.cfg.dataset.center_crop),
            random_flip=bool(self.cfg.dataset.random_flip),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=int(training_cfg.train_batch_size),
            shuffle=True,
            collate_fn=collate_sd3_batches,
            num_workers=int(training_cfg.dataloader_num_workers),
        )
        optimizer = torch.optim.AdamW(
            trainable_parameters(cast(nn.Module, pipe.transformer)),
            lr=float(training_cfg.learning_rate),
        )
        updates_per_epoch = math.ceil(
            len(dataloader) / int(training_cfg.gradient_accumulation_steps)
        )
        initial_step = 0 if resume_checkpoint is None else resume_checkpoint.step
        remaining_steps = int(training_cfg.max_train_steps) - initial_step
        if remaining_steps <= 0:
            final_dir = self.save(accelerator, cast(nn.Module, pipe.transformer), initial_step)
            accelerator.end_training()
            return final_dir
        num_train_epochs = math.ceil(remaining_steps / updates_per_epoch)
        lr_scheduler = get_scheduler(
            str(training_cfg.lr_scheduler),
            optimizer=optimizer,
            num_warmup_steps=int(training_cfg.lr_warmup_steps)
            * int(training_cfg.gradient_accumulation_steps),
            num_training_steps=int(training_cfg.max_train_steps)
            * int(training_cfg.gradient_accumulation_steps),
        )

        prompt_cache = self.build_prompt_cache(
            pipe=pipe,
            prompts=sorted({example.prompt for example in examples}),
            weight_dtype=weight_dtype,
        )
        self.offload_text_encoders(pipe)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pipe.vae.to(accelerator.device, dtype=weight_dtype)
        pipe.transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            pipe.transformer,
            optimizer,
            dataloader,
            lr_scheduler,
        )

        global_step = initial_step
        with make_training_progress(
            accelerator,
            int(training_cfg.max_train_steps),
            global_step,
            f"Training {self.model_key}",
        ) as progress:
            for _epoch in range(num_train_epochs):
                for batch in dataloader:
                    with accelerator.accumulate(pipe.transformer):
                        loss = self.training_step(
                            accelerator=accelerator,
                            batch=batch,
                            pipe=pipe,
                            transformer=cast(nn.Module, pipe.transformer),
                            prompt_cache=prompt_cache,
                            weight_dtype=weight_dtype,
                        )
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                trainable_parameters(cast(nn.Module, pipe.transformer)),
                                float(training_cfg.max_grad_norm),
                            )
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        global_step += 1
                        progress.update(1)
                        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
                        if global_step % int(training_cfg.checkpointing_steps) == 0:
                            self.save(accelerator, cast(nn.Module, pipe.transformer), global_step)
                        if global_step >= int(training_cfg.max_train_steps):
                            break
                if global_step >= int(training_cfg.max_train_steps):
                    break

        final_dir = self.save(accelerator, cast(nn.Module, pipe.transformer), global_step)
        accelerator.end_training()
        return final_dir

    def build_prompt_cache(
        self,
        pipe: Any,
        prompts: list[str],
        weight_dtype: torch.dtype,
    ) -> dict[str, SD3PromptEmbeds]:
        cache: dict[str, SD3PromptEmbeds] = {}
        device = torch.device(str(self.cfg.training.get("text_encoder_device", "cpu")))
        batch_size = int(self.cfg.training.get("prompt_embed_batch_size", 1))
        pipe.text_encoder.to(device=device, dtype=weight_dtype)
        pipe.text_encoder_2.to(device=device, dtype=weight_dtype)
        if pipe.text_encoder_3 is not None:
            pipe.text_encoder_3.to(device=device, dtype=weight_dtype)
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                prompt_batch = prompts[start : start + batch_size]
                prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                    prompt=prompt_batch,
                    prompt_2=None,
                    prompt_3=None,
                    device=device,
                    do_classifier_free_guidance=False,
                    max_sequence_length=int(self.cfg.training.max_sequence_length),
                )
                for index, prompt in enumerate(prompt_batch):
                    cache[prompt] = SD3PromptEmbeds(
                        prompt_embeds=prompt_embeds[index : index + 1].cpu(),
                        pooled_prompt_embeds=pooled_prompt_embeds[index : index + 1].cpu(),
                    )
        return cache

    def offload_text_encoders(self, pipe: Any) -> None:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        if pipe.text_encoder_3 is not None:
            pipe.text_encoder_3.to("cpu")

    def training_step(
        self,
        accelerator: Accelerator,
        batch: SD3PreparedBatch,
        pipe: Any,
        transformer: nn.Module,
        prompt_cache: dict[str, SD3PromptEmbeds],
        weight_dtype: torch.dtype,
    ) -> Tensor:
        target_pixels = batch.target_pixels.to(accelerator.device, dtype=weight_dtype)
        source_pixels = batch.source_pixels.to(accelerator.device, dtype=weight_dtype)

        with torch.no_grad():
            target_encoded = cast(Any, pipe.vae.encode(target_pixels))
            target_latents = cast(Tensor, target_encoded.latent_dist.sample())
            source_encoded = cast(Any, pipe.vae.encode(source_pixels))
            source_latents = cast(Tensor, source_encoded.latent_dist.mode())

        vae_config = cast(Any, pipe.vae.config)
        scaling_factor = float(vae_config.scaling_factor)
        shift_factor = float(vae_config.shift_factor)
        target_latents = (target_latents - shift_factor) * scaling_factor
        source_latents = (source_latents - shift_factor) * scaling_factor
        target_latents = target_latents.to(accelerator.device, dtype=weight_dtype)
        source_latents = source_latents.to(accelerator.device, dtype=weight_dtype)

        noise = torch.randn_like(target_latents)
        batch_size = target_latents.shape[0]
        scheduler = cast(Any, pipe.scheduler)
        timestep_indices = torch.randint(
            0,
            int(scheduler.config.num_train_timesteps),
            (batch_size,),
            device=accelerator.device,
        )
        timesteps = scheduler.timesteps.to(accelerator.device)[timestep_indices]
        sigmas = scheduler.sigmas.to(accelerator.device, dtype=target_latents.dtype)[
            timestep_indices
        ]
        while sigmas.ndim < target_latents.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noisy_target_latents = (1.0 - sigmas) * target_latents + sigmas * noise
        model_input = torch.cat([noisy_target_latents, source_latents], dim=1)

        prompt_embeds = torch.cat(
            [prompt_cache[prompt].prompt_embeds for prompt in batch.prompts], dim=0
        ).to(accelerator.device, dtype=weight_dtype)
        pooled_prompt_embeds = torch.cat(
            [prompt_cache[prompt].pooled_prompt_embeds for prompt in batch.prompts], dim=0
        ).to(accelerator.device, dtype=weight_dtype)

        model_pred = cast(
            tuple[Tensor],
            cast(Any, transformer)(
                hidden_states=model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            ),
        )[0]
        target = noise - target_latents
        return cast(Tensor, F.mse_loss(model_pred.float(), target.float(), reduction="mean"))

    def save(self, accelerator: Accelerator, transformer: nn.Module, step: int) -> Path:
        if not accelerator.is_main_process:
            return self.output_dir
        save_dir = self.output_dir / f"checkpoint-{step:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        cast(Any, unwrapped_transformer).save_lora_adapter(
            str(save_dir),
            adapter_name="default",
            safe_serialization=True,
        )
        projection = cast(nn.Conv2d, cast(Any, unwrapped_transformer).pos_embed.proj)
        save_file(
            {
                "weight": projection.weight.detach().cpu(),
                "bias": (
                    projection.bias.detach().cpu()
                    if projection.bias is not None
                    else torch.empty(0)
                ),
            },
            save_dir / "sd3_input_projection_32ch.safetensors",
            metadata={"format": "pt", "source": "local_edit_lora"},
        )
        manifest = {
            "model_key": self.model_key,
            "model_id": str(self.model_cfg.pretrained_model_name_or_path),
            "trainer": str(self.model_cfg.trainer),
            "step": step,
            "requires_source_image": True,
            "conditioning": "source_latents_concatenated_to_noisy_target_latents",
            "lora_rank": int(self.cfg.training.rank),
            "lora_alpha": int(self.cfg.training.lora_alpha),
            "dataset_train_dir": str(self.cfg.dataset.train_dir),
            "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        }
        with (save_dir / "training_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return save_dir


def load_sd3_pipeline(
    cfg: DictConfig,
    model_key: str,
    checkpoint_dir: Path,
) -> DiffusionPipeline:
    model_cfg = cfg.models[model_key]
    dtype = dtype_from_precision(str(cfg.evaluation.mixed_precision))
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": bool(cfg.evaluation.get("local_files_only", False)),
    }
    if bool(cfg.evaluation.get("sd3_disable_t5", False)):
        load_kwargs["text_encoder_3"] = None
        load_kwargs["tokenizer_3"] = None
    revision = none_if_null(model_cfg.get("revision"))
    variant = none_if_null(model_cfg.get("variant"))
    if revision is not None:
        load_kwargs["revision"] = revision
    if variant is not None:
        load_kwargs["variant"] = variant
    pipe = DiffusionPipeline.from_pretrained(
        str(model_cfg.pretrained_model_name_or_path),
        **load_kwargs,
    )
    expand_sd3_transformer_input_for_paired_edit(cast(nn.Module, pipe.transformer))
    apply_sd3_input_projection_patch(cast(nn.Module, pipe.transformer), checkpoint_dir)
    cast(Any, pipe.transformer).load_lora_adapter(
        str(checkpoint_dir),
        prefix=None,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="aura",
    )
    cast(Any, pipe.transformer).set_adapters(["aura"], weights=[float(cfg.evaluation.lora_scale)])
    return pipe


def load_sd3_condition_image(path: Path, width: int, height: int) -> PILImage:
    image = load_rgb_image(path)
    return cast(
        PILImage,
        ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
        ),
    )


def encode_sd3_image_latents(
    pipe: DiffusionPipeline,
    image: PILImage,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    width: int,
    height: int,
) -> Tensor:
    image_tensor = cast(Any, pipe.image_processor).preprocess(image, height=height, width=width)
    image_tensor = image_tensor.to(device=device, dtype=dtype)
    encoded = cast(Any, pipe.vae.encode(image_tensor))
    latents = cast(Tensor, encoded.latent_dist.sample(generator=generator))
    vae_config = cast(Any, pipe.vae.config)
    latents = (latents - float(vae_config.shift_factor)) * float(vae_config.scaling_factor)
    return latents.to(device=device, dtype=dtype)


def sd3_timesteps(
    pipe: DiffusionPipeline,
    height: int,
    width: int,
    num_inference_steps: int,
    device: torch.device,
) -> Tensor:
    scheduler_kwargs: dict[str, Any] = {}
    scheduler = cast(Any, pipe.scheduler)
    if bool(scheduler.config.get("use_dynamic_shifting", False)):
        image_seq_len = (
            height // cast(Any, pipe).vae_scale_factor // pipe.transformer.config.patch_size
        ) * (width // cast(Any, pipe).vae_scale_factor // pipe.transformer.config.patch_size)
        scheduler_kwargs["mu"] = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.16),
        )
    timesteps, _ = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        **scheduler_kwargs,
    )
    return cast(Tensor, timesteps)


def decode_sd3_latents(pipe: DiffusionPipeline, latents: Tensor) -> PILImage:
    vae_config = cast(Any, pipe.vae.config)
    decode_latents = (latents / float(vae_config.scaling_factor)) + float(vae_config.shift_factor)
    image = pipe.vae.decode(decode_latents, return_dict=False)[0]
    images = cast(
        list[PILImage], cast(Any, pipe.image_processor).postprocess(image, output_type="pil")
    )
    if len(images) != 1:
        raise RuntimeError(f"Expected 1 SD3 output image, received {len(images)}.")
    return images[0]


def run_sd3_batch(
    pipe: DiffusionPipeline,
    cfg: DictConfig,
    input_paths: list[Path],
    device_name: str,
    batch_offset: int,
) -> list[PILImage]:
    images: list[PILImage] = []
    device = torch.device(device_name)
    dtype = dtype_from_precision(str(cfg.evaluation.mixed_precision))
    configured_seed = evaluation_seed(cfg)
    if configured_seed is None:
        raise ValueError("SD3 inference requires evaluation.seed to be configured.")
    seed = configured_seed
    width = int(cfg.evaluation.width)
    height = int(cfg.evaluation.height)
    num_inference_steps = int(cfg.evaluation.num_inference_steps)
    guidance_scale = float(cfg.evaluation.guidance_scale)
    do_classifier_free_guidance = guidance_scale > 1.0

    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = cast(Any, pipe).encode_prompt(
            prompt=str(cfg.evaluation.prompt),
            prompt_2=None,
            prompt_3=None,
            negative_prompt="",
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=do_classifier_free_guidance,
            device=device,
            max_sequence_length=int(cfg.evaluation.max_sequence_length),
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)

        for offset, input_path in enumerate(input_paths):
            timesteps = sd3_timesteps(pipe, height, width, num_inference_steps, device)
            start_timestep = timesteps[:1]
            generator = torch.Generator(device=device_name).manual_seed(
                seed + batch_offset + offset
            )
            source_image = load_sd3_condition_image(input_path, width, height)
            source_latents = encode_sd3_image_latents(
                pipe,
                source_image,
                device,
                dtype,
                generator,
                width,
                height,
            )
            noise = torch.randn(
                source_latents.shape,
                generator=generator,
                device=device,
                dtype=source_latents.dtype,
            )
            latents = pipe.scheduler.scale_noise(source_latents, start_timestep, noise)

            for timestep in timesteps:
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                source_model_input = (
                    torch.cat([source_latents] * 2)
                    if do_classifier_free_guidance
                    else source_latents
                )
                model_input = torch.cat([latent_model_input, source_model_input], dim=1)
                timestep_batch = timestep.expand(model_input.shape[0])
                noise_pred = cast(Any, pipe.transformer)(
                    hidden_states=model_input,
                    timestep=timestep_batch,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

            images.append(decode_sd3_latents(pipe, latents))
    return images
