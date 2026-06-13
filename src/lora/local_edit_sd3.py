from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from omegaconf import DictConfig
from PIL import Image
from peft import LoraConfig
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from lora.local_edit_common import (
    PairExample,
    apply_lora_checkpoint,
    dtype_from_precision,
    freeze_module,
    load_pair_examples,
    load_rgb_image,
    make_training_progress,
    none_if_null,
    resolve_resume_checkpoint,
    resolve_repo_path,
    trainable_parameters,
)


@dataclass(frozen=True)
class SD3PreparedBatch:
    source_pixels: Tensor
    target_pixels: Tensor
    prompts: list[str]


@dataclass(frozen=True)
class SD3PromptEmbeds:
    prompt_embeds: Tensor
    pooled_prompt_embeds: Tensor


class SD3PairedEditDataset(Dataset[SD3PreparedBatch]):
    def __init__(
        self,
        examples: list[PairExample],
        resolution: int,
        center_crop: bool,
        random_flip: bool,
    ) -> None:
        self.examples = examples
        crop: transforms.CenterCrop | transforms.RandomCrop
        crop = (
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        )
        transform_steps: list[Any] = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            crop,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        self.image_transform = transforms.Compose(transform_steps)
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SD3PreparedBatch:
        example = self.examples[index]
        source_image = load_rgb_image(example.source_path)
        target_image = load_rgb_image(example.target_path)
        if self.random_flip and random.random() < 0.5:
            source_image = source_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            target_image = target_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return SD3PreparedBatch(
            source_pixels=cast(Tensor, self.image_transform(source_image)),
            target_pixels=cast(Tensor, self.image_transform(target_image)),
            prompts=[example.prompt],
        )


def collate_sd3_batches(items: list[SD3PreparedBatch]) -> SD3PreparedBatch:
    return SD3PreparedBatch(
        source_pixels=torch.stack([item.source_pixels for item in items]),
        target_pixels=torch.stack([item.target_pixels for item in items]),
        prompts=[item.prompts[0] for item in items],
    )


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
        freeze_module(pipe.text_encoder_3)
        freeze_module(pipe.transformer)
        pipe.vae.eval()
        pipe.text_encoder.eval()
        pipe.text_encoder_2.eval()
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
        input_ids = batch.input_ids.to(accelerator.device)
        
        target_encode = self.encode(target_pixels)
        noisy_latent = (1 - sigma) * target_encode + sigma * noise
        source_encode = self.encode(source_pixels)
        input = torch.concat([source_encode, target_encode], dim=1)
        timestamp = torch.rand(0, 1)
        pred = pipe(input, noisy_latent, timestamp)
        return mse.loss(target_encode - pred, target_encode)

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
