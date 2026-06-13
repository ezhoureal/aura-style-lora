from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from tqdm.std import tqdm as Tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_IMAGE_SUFFIXES = {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp"}
CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint-(?P<step>\d+)$")


@dataclass(frozen=True)
class PairExample:
    source_path: Path
    target_path: Path
    prompt: str


@dataclass(frozen=True)
class ResumeCheckpoint:
    path: Path
    step: int


@dataclass(frozen=True)
class PairedPromptBatch:
    source_pixels: Tensor
    target_pixels: Tensor
    prompts: list[str]


class EditTrainer(Protocol):
    def train(self) -> Path: ...


class ProgressAccelerator(Protocol):
    @property
    def is_local_main_process(self) -> bool: ...


def make_training_progress(
    accelerator: ProgressAccelerator, total_steps: int, initial_step: int, description: str
) -> Tqdm:
    return tqdm(
        total=total_steps,
        initial=initial_step,
        desc=description,
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )


def load_rgb_image(path: Path) -> PILImage:
    with Image.open(path) as image:
        transposed = cast(PILImage, ImageOps.exif_transpose(image))
        return transposed.convert("RGB")


def make_pair_image_transform(resolution: int, center_crop: bool) -> transforms.Compose:
    crop: transforms.CenterCrop | transforms.RandomCrop
    crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
    transform_steps: list[Any] = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        crop,
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    return transforms.Compose(transform_steps)


def load_transformed_pair(
    example: PairExample,
    image_transform: transforms.Compose,
    random_flip: bool,
) -> tuple[Tensor, Tensor]:
    source_image = load_rgb_image(example.source_path)
    target_image = load_rgb_image(example.target_path)
    if random_flip and random.random() < 0.5:
        source_image = source_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        target_image = target_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return (
        cast(Tensor, image_transform(source_image)),
        cast(Tensor, image_transform(target_image)),
    )


class PairedPromptDataset(Dataset[PairedPromptBatch]):
    def __init__(
        self,
        examples: list[PairExample],
        resolution: int,
        center_crop: bool,
        random_flip: bool,
    ) -> None:
        self.examples = examples
        self.image_transform = make_pair_image_transform(resolution, center_crop)
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PairedPromptBatch:
        example = self.examples[index]
        source_pixels, target_pixels = load_transformed_pair(
            example,
            self.image_transform,
            self.random_flip,
        )
        return PairedPromptBatch(
            source_pixels=source_pixels,
            target_pixels=target_pixels,
            prompts=[example.prompt],
        )


def collate_paired_prompt_batches(items: list[PairedPromptBatch]) -> PairedPromptBatch:
    return PairedPromptBatch(
        source_pixels=torch.stack([item.source_pixels for item in items]),
        target_pixels=torch.stack([item.target_pixels for item in items]),
        prompts=[item.prompts[0] for item in items],
    )


def read_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checkpoint_step_from_path(path: Path) -> int:
    manifest_path = path / "training_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        step = manifest.get("step")
        if isinstance(step, int):
            return step
        raise ValueError(f"Checkpoint manifest has invalid step: {manifest_path}")

    match = CHECKPOINT_DIR_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Cannot infer checkpoint step from path: {path}")
    return int(match.group("step"))


def latest_checkpoint(output_dir: Path) -> ResumeCheckpoint | None:
    if not output_dir.exists():
        return None
    checkpoints: list[ResumeCheckpoint] = []
    for path in output_dir.iterdir():
        if not path.is_dir() or CHECKPOINT_DIR_PATTERN.match(path.name) is None:
            continue
        checkpoints.append(ResumeCheckpoint(path=path, step=checkpoint_step_from_path(path)))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda checkpoint: checkpoint.step)


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_resume_checkpoint(
    training_cfg: DictConfig, output_dir: Path
) -> ResumeCheckpoint | None:
    configured = training_cfg.get("resume_from_checkpoint")
    if configured is None:
        return None
    value = str(configured)
    if value.lower() in {"", "false", "none", "null"}:
        return None
    if value == "latest":
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoints found to resume under: {output_dir}")
        return checkpoint

    checkpoint_path = resolve_repo_path(value)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint directory does not exist: {checkpoint_path}")
    if not checkpoint_path.is_dir():
        raise NotADirectoryError(f"Resume checkpoint is not a directory: {checkpoint_path}")
    return ResumeCheckpoint(
        path=checkpoint_path,
        step=checkpoint_step_from_path(checkpoint_path),
    )


def apply_lora_checkpoint(module: nn.Module, checkpoint_dir: Path) -> None:
    weights_path = checkpoint_dir / "pytorch_lora_weights.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing LoRA checkpoint weights: {weights_path}")
    state = load_file(weights_path)
    set_peft_model_state_dict(module, state, adapter_name="default")


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


def discover_images(input_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if limit is None:
        return paths
    return paths[:limit]


def batched_paths(paths: list[Path], batch_size: int) -> list[list[Path]]:
    if batch_size < 1:
        raise ValueError(f"Batch size must be at least 1, received {batch_size}")
    return [paths[index : index + batch_size] for index in range(0, len(paths), batch_size)]


def generators_for_batch(
    seed: int | None,
    device_name: str,
    batch_offset: int,
    batch_size: int,
) -> torch.Generator | list[torch.Generator] | None:
    if seed is None:
        return None
    if batch_size == 1:
        return torch.Generator(device=device_name).manual_seed(seed + batch_offset)
    return [
        torch.Generator(device=device_name).manual_seed(seed + batch_offset + index)
        for index in range(batch_size)
    ]


def evaluation_seed(cfg: DictConfig) -> int | None:
    configured_seed = none_if_null(cfg.evaluation.seed)
    if configured_seed is None:
        return None
    return int(configured_seed)
