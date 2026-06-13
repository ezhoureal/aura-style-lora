from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from omegaconf import DictConfig
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
from peft import LoraConfig
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
    generators_for_batch,
    load_pair_examples,
    load_rgb_image,
    make_training_progress,
    none_if_null,
    resolve_resume_checkpoint,
    resolve_repo_path,
    trainable_parameters,
)


FluxPreparedBatch = PairedPromptBatch
FluxPairedEditDataset = PairedPromptDataset
collate_flux_batches = collate_paired_prompt_batches


def flow_match_noisy_latents(clean_latents: Tensor, noise: Tensor, sigmas: Tensor) -> Tensor:
    while sigmas.ndim < clean_latents.ndim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas * noise + (1.0 - sigmas) * clean_latents


def flow_match_training_target(clean_latents: Tensor, noise: Tensor) -> Tensor:
    return noise - clean_latents


class Flux2PairedEditLoraTrainer:
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

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True,
            local_files_only=bool(training_cfg.get("local_files_only", False)),
        )
        freeze_module(pipe.vae)
        freeze_module(pipe.text_encoder)
        freeze_module(pipe.transformer)
        pipe.vae.eval()
        pipe.text_encoder.eval()
        pipe.transformer.train()

        pipe.transformer.add_adapter(
            LoraConfig(
                r=int(training_cfg.rank),
                lora_alpha=int(training_cfg.lora_alpha),
                init_lora_weights="gaussian",
                target_modules=list(training_cfg.flux2_target_modules),
            )
        )
        if bool(training_cfg.gradient_checkpointing):
            pipe.transformer.enable_gradient_checkpointing()

        resume_checkpoint = resolve_resume_checkpoint(training_cfg, self.output_dir)
        if resume_checkpoint is not None:
            apply_lora_checkpoint(pipe.transformer, resume_checkpoint.path)

        examples = load_pair_examples(self.cfg.dataset)
        dataset = FluxPairedEditDataset(
            examples=examples,
            resolution=int(self.cfg.dataset.resolution),
            center_crop=bool(self.cfg.dataset.center_crop),
            random_flip=bool(self.cfg.dataset.random_flip),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=int(training_cfg.train_batch_size),
            shuffle=True,
            collate_fn=collate_flux_batches,
            num_workers=int(training_cfg.dataloader_num_workers),
        )
        optimizer = torch.optim.AdamW(
            trainable_parameters(pipe.transformer), lr=float(training_cfg.learning_rate)
        )
        updates_per_epoch = math.ceil(
            len(dataloader) / int(training_cfg.gradient_accumulation_steps)
        )
        initial_step = 0 if resume_checkpoint is None else resume_checkpoint.step
        remaining_steps = int(training_cfg.max_train_steps) - initial_step
        if remaining_steps <= 0:
            final_dir = self.save(
                accelerator, cast(Flux2Transformer2DModel, pipe.transformer), initial_step
            )
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
        pipe.text_encoder.to("cpu")  # save VRAM
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
                            transformer=cast(Flux2Transformer2DModel, pipe.transformer),
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
                            self.save(
                                accelerator,
                                cast(Flux2Transformer2DModel, pipe.transformer),
                                global_step,
                            )
                        if global_step >= int(training_cfg.max_train_steps):
                            break
                if global_step >= int(training_cfg.max_train_steps):
                    break

        final_dir = self.save(
            accelerator, cast(Flux2Transformer2DModel, pipe.transformer), global_step
        )
        accelerator.end_training()
        return final_dir

    def build_prompt_cache(
        self,
        pipe: Any,
        prompts: list[str],
        weight_dtype: torch.dtype,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        cache: dict[str, tuple[Tensor, Tensor]] = {}
        device = torch.device(str(self.cfg.training.get("text_encoder_device", "cpu")))
        batch_size = int(self.cfg.training.get("prompt_embed_batch_size", 4))
        pipe.text_encoder.to(device=device, dtype=weight_dtype)
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                prompt_batch = prompts[start : start + batch_size]
                prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=prompt_batch,
                    device=device,
                    max_sequence_length=int(self.cfg.training.max_sequence_length),
                    text_encoder_out_layers=cast(
                        Any, tuple(int(v) for v in self.cfg.training.text_encoder_out_layers)
                    ),
                )
                for index, prompt in enumerate(prompt_batch):
                    cache[prompt] = (
                        prompt_embeds[index : index + 1].cpu(),
                        text_ids[index : index + 1].cpu(),
                    )
        return cache

    def training_step(
        self,
        accelerator: Accelerator,
        batch: FluxPreparedBatch,
        pipe: Any,
        transformer: Flux2Transformer2DModel,
        prompt_cache: dict[str, tuple[Tensor, Tensor]],
        weight_dtype: torch.dtype,
    ) -> Tensor:
        target_pixels = batch.target_pixels.to(accelerator.device, dtype=weight_dtype)
        source_pixels = batch.source_pixels.to(accelerator.device, dtype=weight_dtype)
        generator = torch.Generator(device=accelerator.device)
        target_grid_latents = cast(Any, pipe)._encode_vae_image(target_pixels, generator=generator)
        source_grid_latents = cast(Any, pipe)._encode_vae_image(source_pixels, generator=generator)
        latent_ids = cast(Any, pipe)._prepare_latent_ids(target_grid_latents).to(accelerator.device)
        source_ids = cast(Any, pipe)._prepare_image_ids([source_grid_latents[:1]])
        source_ids = source_ids.repeat(target_grid_latents.shape[0], 1, 1).to(accelerator.device)
        target_latents = cast(Any, pipe)._pack_latents(target_grid_latents)
        source_latents = cast(Any, pipe)._pack_latents(source_grid_latents)

        prompt_embeds = torch.cat([prompt_cache[prompt][0] for prompt in batch.prompts], dim=0)
        text_ids = torch.cat([prompt_cache[prompt][1] for prompt in batch.prompts], dim=0)
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        text_ids = text_ids.to(accelerator.device)

        noise = torch.randn_like(target_latents)
        sigmas = torch.rand(
            target_latents.shape[0], device=target_latents.device, dtype=target_latents.dtype
        )
        noisy_latents = flow_match_noisy_latents(target_latents, noise, sigmas)
        target = flow_match_training_target(target_latents, noise)
        model_input = torch.cat([noisy_latents, source_latents], dim=1)
        img_ids = torch.cat([latent_ids, source_ids], dim=1)
        guidance = torch.full(
            (target_latents.shape[0],),
            float(self.cfg.training.guidance_scale),
            device=accelerator.device,
            dtype=torch.float32,
        )
        model_pred = transformer(
            hidden_states=model_input.to(weight_dtype),
            timestep=sigmas,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        model_pred = model_pred[:, : target_latents.shape[1]]
        return cast(Tensor, F.mse_loss(model_pred.float(), target.float(), reduction="mean"))

    def save(
        self, accelerator: Accelerator, transformer: Flux2Transformer2DModel, step: int
    ) -> Path:
        if not accelerator.is_main_process:
            return self.output_dir
        save_dir = self.output_dir / f"checkpoint-{step:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        cast(Flux2Transformer2DModel, unwrapped_transformer).save_lora_adapter(
            str(save_dir),
            adapter_name="default",
            safe_serialization=True,
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


def load_flux2_pipeline(
    cfg: DictConfig,
    model_key: str,
    checkpoint_dir: Path,
) -> DiffusionPipeline:
    model_cfg = cfg.models[model_key]
    dtype = dtype_from_precision(str(cfg.evaluation.mixed_precision))
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    revision = none_if_null(model_cfg.get("revision"))
    variant = none_if_null(model_cfg.get("variant"))
    if revision is not None:
        load_kwargs["revision"] = revision
    if variant is not None:
        load_kwargs["variant"] = variant
    load_kwargs["low_cpu_mem_usage"] = True
    load_kwargs["local_files_only"] = bool(cfg.evaluation.get("local_files_only", False))
    pipe = DiffusionPipeline.from_pretrained(
        str(model_cfg.pretrained_model_name_or_path),
        **load_kwargs,
    )
    cast(Any, pipe.transformer).load_lora_adapter(
        str(checkpoint_dir),
        prefix=None,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="aura",
    )
    cast(Any, pipe.transformer).set_adapters(["aura"], weights=[float(cfg.evaluation.lora_scale)])
    return pipe


def load_flux2_condition_image(path: Path, width: int, height: int) -> PILImage:
    image = load_rgb_image(path)
    return cast(
        PILImage,
        ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
        ),
    )


def run_flux2_batch(
    pipe: DiffusionPipeline,
    cfg: DictConfig,
    input_paths: list[Path],
    device_name: str,
    batch_offset: int,
) -> list[PILImage]:
    images: list[PILImage] = []
    seed = evaluation_seed(cfg)
    # no real batching. Diffuser APi doesn't support it
    for offset, input_path in enumerate(input_paths):
        width = int(cfg.evaluation.width)
        height = int(cfg.evaluation.height)
        call_kwargs: dict[str, Any] = {
            "prompt": str(cfg.evaluation.prompt),
            "image": load_flux2_condition_image(input_path, width, height),
            "height": height,
            "width": width,
            "num_inference_steps": int(cfg.evaluation.num_inference_steps),
            "guidance_scale": float(cfg.evaluation.guidance_scale),
            "generator": generators_for_batch(
                seed,
                device_name,
                batch_offset + offset,
                1,
            ),
            "max_sequence_length": int(cfg.evaluation.max_sequence_length),
            "text_encoder_out_layers": tuple(
                int(value) for value in cfg.evaluation.text_encoder_out_layers
            ),
        }
        result = cast(Any, pipe)(**call_kwargs)
        result_images = cast(list[PILImage], result.images)
        if len(result_images) != 1:
            raise RuntimeError(
                f"Expected 1 output for {input_path.name}, received {len(result_images)}."
            )
        images.append(result_images[0])
    return images
