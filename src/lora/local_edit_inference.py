from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
)
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
from safetensors.torch import load_file

from lora.local_edit_training import (
    REPO_ROOT,
    SUPPORTED_IMAGE_SUFFIXES,
    dtype_from_precision,
    expand_unet_conv_in_for_ip2p,
    none_if_null,
    resolve_repo_path,
    selected_model_keys,
)


def discover_images(input_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if limit is None:
        return paths
    return paths[:limit]


def load_rgb_image(path: Path) -> PILImage:
    with Image.open(path) as image:
        transposed = cast(PILImage, ImageOps.exif_transpose(image))
        return transposed.convert("RGB")


def checkpoint_dir_for_model(cfg: DictConfig, model_key: str) -> Path:
    configured = cfg.evaluation.checkpoint_dir
    if configured is not None:
        return resolve_repo_path(str(configured))
    output_root = resolve_repo_path(str(cfg.training.output_root))
    return output_root / model_key / f"checkpoint-{int(cfg.training.max_train_steps):06d}"


def load_kwargs_for_model(model_cfg: DictConfig, dtype: torch.dtype) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    revision = none_if_null(model_cfg.get("revision"))
    variant = none_if_null(model_cfg.get("variant"))
    if revision is not None:
        kwargs["revision"] = revision
    if variant is not None:
        kwargs["variant"] = variant
    return kwargs


def apply_conv_in_patch(pipe: StableDiffusionInstructPix2PixPipeline, checkpoint_dir: Path) -> None:
    patch_path = checkpoint_dir / "unet_conv_in_8ch.safetensors"
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing image-edit conv_in patch: {patch_path}")
    state = load_file(patch_path)
    with torch.no_grad():
        pipe.unet.conv_in.weight.copy_(state["weight"].to(pipe.unet.conv_in.weight.device))
        bias = state["bias"]
        if pipe.unet.conv_in.bias is not None and bias.numel() > 0:
            pipe.unet.conv_in.bias.copy_(bias.to(pipe.unet.conv_in.bias.device))


def load_pipeline(
    cfg: DictConfig,
    model_key: str,
    checkpoint_dir: Path,
) -> StableDiffusionInstructPix2PixPipeline:
    model_cfg = cfg.models[model_key]
    if str(model_cfg.trainer) != "stable_diffusion_ip2p_lora":
        raise NotImplementedError(
            f"Local inference is not implemented for trainer {model_cfg.trainer}"
        )
    dtype = dtype_from_precision(str(cfg.evaluation.mixed_precision))
    model_id = str(model_cfg.pretrained_model_name_or_path)
    load_kwargs = load_kwargs_for_model(model_cfg, dtype)
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


def run_model(cfg: DictConfig, model_key: str) -> list[Path]:
    checkpoint_dir = checkpoint_dir_for_model(cfg, model_key)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    input_dir = resolve_repo_path(str(cfg.evaluation.input_dir))
    configured_limit = none_if_null(cfg.evaluation.limit)
    limit = int(configured_limit) if configured_limit is not None else None
    image_paths = discover_images(input_dir, limit)
    if not image_paths:
        raise ValueError(f"No supported eval images found in {input_dir}")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device_name).manual_seed(int(cfg.evaluation.seed))
    pipe = load_pipeline(cfg, model_key, checkpoint_dir)
    pipe.to(device_name)
    output_dir = resolve_repo_path(str(cfg.evaluation.output_root)) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for index, input_path in enumerate(image_paths, start=1):
        result = cast(
            Any,
            pipe(
                prompt=str(cfg.evaluation.prompt),
                image=load_rgb_image(input_path),
                num_inference_steps=int(cfg.evaluation.num_inference_steps),
                guidance_scale=float(cfg.evaluation.guidance_scale),
                image_guidance_scale=float(cfg.evaluation.image_guidance_scale),
                generator=generator,
            ),
        )
        output_path = output_dir / f"{input_path.stem}-{index:03d}.png"
        result.images[0].save(output_path)
        output_paths.append(output_path)
    return output_paths


def configure_environment(cfg: DictConfig) -> None:
    os.environ.setdefault("HF_ENDPOINT", str(cfg.environment.hf_endpoint))
    os.environ.setdefault(
        "HF_HUB_ENABLE_HF_TRANSFER", str(cfg.environment.hf_hub_enable_hf_transfer)
    )


def run(cfg: DictConfig) -> None:
    configure_environment(cfg)
    report: dict[str, list[str]] = {}
    for model_key in selected_model_keys(cfg):
        report[model_key] = [str(path) for path in run_model(cfg, model_key)]
    print(json.dumps(report, indent=2))


def main() -> None:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="local_edit_lora", overrides=sys.argv[1:])
    run(cfg)
