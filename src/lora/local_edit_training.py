from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from hydra import compose, initialize_config_dir
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import DictConfig, ListConfig, OmegaConf
from peft import LoraConfig
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_IMAGE_SUFFIXES = {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class PairExample:
    source_path: Path
    target_path: Path
    prompt: str


@dataclass(frozen=True)
class PreparedBatch:
    source_pixels: Tensor
    target_pixels: Tensor
    input_ids: Tensor


class EditTrainer(Protocol):
    def train(self) -> Path: ...


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

    def __getitem__(self, index: int) -> PreparedBatch:
        example = self.examples[index]
        source_image = load_rgb_image(example.source_path)
        target_image = load_rgb_image(example.target_path)
        if self.random_flip and random.random() < 0.5:
            source_image = source_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            target_image = target_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        tokens = self.tokenizer(
            example.prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return PreparedBatch(
            source_pixels=cast(Tensor, self.image_transform(source_image)),
            target_pixels=cast(Tensor, self.image_transform(target_image)),
            input_ids=cast(Tensor, tokens.input_ids[0]),
        )


def collate_batches(items: list[PreparedBatch]) -> PreparedBatch:
    return PreparedBatch(
        source_pixels=torch.stack([item.source_pixels for item in items]),
        target_pixels=torch.stack([item.target_pixels for item in items]),
        input_ids=torch.stack([item.input_ids for item in items]),
    )


def load_rgb_image(path: Path) -> PILImage:
    with Image.open(path) as image:
        transposed = cast(PILImage, ImageOps.exif_transpose(image))
        return transposed.convert("RGB")


def read_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_relative_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_pair_examples(dataset_cfg: DictConfig) -> list[PairExample]:
    train_dir = resolve_repo_path(str(dataset_cfg.train_dir))
    metadata_path = train_dir / "metadata.jsonl"
    rows = read_metadata(metadata_path)
    examples: list[PairExample] = []

    for row in rows:
        if row.get(str(dataset_cfg.kind_key)) != str(dataset_cfg.paired_kind):
            continue
        image_name = row.get(str(dataset_cfg.image_key))
        conditioning_name = row.get(str(dataset_cfg.conditioning_key))
        prompt = row.get(str(dataset_cfg.prompt_key))
        if not isinstance(image_name, str) or not isinstance(conditioning_name, str):
            continue
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        target_path = train_dir / image_name
        source_path = resolve_relative_path(train_dir, conditioning_name)
        if target_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if source_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target image: {target_path}")
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source conditioning image: {source_path}")
        examples.append(
            PairExample(source_path=source_path, target_path=target_path, prompt=prompt)
        )

    if not examples:
        raise ValueError(f"No paired image-edit examples found in {metadata_path}")
    return examples


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


def freeze_module(module: nn.Module) -> None:
    module.requires_grad_(False)


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def dtype_from_precision(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value in {"no", "fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported mixed precision value: {value}")


def none_if_null(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() == "null":
        return None
    return text


class StableDiffusionIp2PLoraTrainer:
    def __init__(self, cfg: DictConfig, model_key: str) -> None:
        self.cfg = cfg
        self.model_key = model_key
        self.model_cfg = cfg.models[model_key]
        output_root = resolve_repo_path(str(cfg.training.output_root))
        self.output_dir = output_root / model_key

    def train(self) -> Path:
        training_cfg = self.cfg.training
        model_id = str(self.model_cfg.pretrained_model_name_or_path)
        revision = none_if_null(self.model_cfg.get("revision"))
        variant = none_if_null(self.model_cfg.get("variant"))
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

        load_kwargs: dict[str, Any] = {}
        if revision is not None:
            load_kwargs["revision"] = revision
        if variant is not None:
            load_kwargs["variant"] = variant
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
        unet.conv_in.requires_grad_(True)
        unet.add_adapter(
            LoraConfig(
                r=int(training_cfg.rank),
                lora_alpha=int(training_cfg.lora_alpha),
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
        )
        if bool(training_cfg.gradient_checkpointing):
            unet.enable_gradient_checkpointing()

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
        optimizer = torch.optim.AdamW(
            trainable_parameters(unet), lr=float(training_cfg.learning_rate)
        )
        updates_per_epoch = math.ceil(
            len(dataloader) / int(training_cfg.gradient_accumulation_steps)
        )
        num_train_epochs = math.ceil(int(training_cfg.max_train_steps) / updates_per_epoch)
        lr_scheduler = get_scheduler(
            str(training_cfg.lr_scheduler),
            optimizer=optimizer,
            num_warmup_steps=int(training_cfg.lr_warmup_steps)
            * int(training_cfg.gradient_accumulation_steps),
            num_training_steps=int(training_cfg.max_train_steps)
            * int(training_cfg.gradient_accumulation_steps),
        )

        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet,
            optimizer,
            dataloader,
            lr_scheduler,
        )
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        global_step = 0
        unet.train()

        for _epoch in range(num_train_epochs):
            for batch in dataloader:
                with accelerator.accumulate(unet):
                    loss = self.training_step(
                        accelerator=accelerator,
                        batch=batch,
                        vae=vae,
                        text_encoder=text_encoder,
                        unet=cast(UNet2DConditionModel, unet),
                        noise_scheduler=noise_scheduler,
                        weight_dtype=weight_dtype,
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            trainable_parameters(cast(nn.Module, unet)),
                            float(training_cfg.max_grad_norm),
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % int(training_cfg.checkpointing_steps) == 0:
                        self.save(accelerator, cast(UNet2DConditionModel, unet), global_step)
                    if global_step >= int(training_cfg.max_train_steps):
                        break
            if global_step >= int(training_cfg.max_train_steps):
                break

        final_dir = self.save(accelerator, cast(UNet2DConditionModel, unet), global_step)
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
        save_dir = self.output_dir / f"checkpoint-{step:06d}"
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


class UnsupportedLocalEditTrainer:
    def __init__(self, cfg: DictConfig, model_key: str) -> None:
        self.cfg = cfg
        self.model_key = model_key
        self.model_cfg = cfg.models[model_key]

    def train(self) -> Path:
        display_name = str(self.model_cfg.display_name)
        note = str(self.model_cfg.get("note", "No local paired edit trainer is configured."))
        raise NotImplementedError(f"{display_name}: {note}")


def make_trainer(cfg: DictConfig, model_key: str) -> EditTrainer:
    trainer_name = str(cfg.models[model_key].trainer)
    if trainer_name == "stable_diffusion_ip2p_lora":
        return StableDiffusionIp2PLoraTrainer(cfg, model_key)
    if trainer_name == "unsupported_local_edit":
        return UnsupportedLocalEditTrainer(cfg, model_key)
    raise ValueError(f"Unknown trainer adapter: {trainer_name}")


def configure_environment(cfg: DictConfig) -> None:
    os.environ.setdefault("HF_ENDPOINT", str(cfg.environment.hf_endpoint))
    os.environ.setdefault(
        "HF_HUB_ENABLE_HF_TRANSFER", str(cfg.environment.hf_hub_enable_hf_transfer)
    )


def selected_model_keys(cfg: DictConfig) -> list[str]:
    selected_models = OmegaConf.select(cfg, "selected_models")
    if selected_models is None:
        return [str(model_key) for model_key in cfg.models.keys()]
    if isinstance(selected_models, ListConfig):
        return [str(model_key) for model_key in selected_models]
    return [str(selected_models)]


def run(cfg: DictConfig) -> None:
    configure_environment(cfg)
    for model_key in selected_model_keys(cfg):
        trainer = make_trainer(cfg, model_key)
        output_dir = trainer.train()
        print(json.dumps({"model_key": model_key, "output_dir": str(output_dir)}, indent=2))


def main() -> None:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="local_edit_lora", overrides=sys.argv[1:])
    run(cfg)
