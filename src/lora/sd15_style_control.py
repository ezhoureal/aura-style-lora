from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import (
    StableDiffusionControlNetImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import ImageFilter, ImageOps
from PIL.Image import Image as PILImage

from lora.local_edit_common import (
    REPO_ROOT,
    SUPPORTED_IMAGE_SUFFIXES,
    dtype_from_precision,
    generators_for_batch,
    load_condition_image,
    load_rgb_image,
    none_if_null,
    read_metadata,
    resolve_repo_path,
)


@dataclass(frozen=True)
class Stage0Example:
    image_path: Path
    caption: str


@dataclass(frozen=True)
class ControlRuntime:
    name: str
    scale: float
    start: float
    end: float
    preprocessor: str
    precomputed_dir: Path | None
    threshold: int


def configure_environment(cfg: DictConfig) -> None:
    os.environ.setdefault("HF_ENDPOINT", str(cfg.environment.hf_endpoint))
    os.environ.setdefault(
        "HF_HUB_ENABLE_HF_TRANSFER", str(cfg.environment.hf_hub_enable_hf_transfer)
    )


def selected_baseline_keys(cfg: DictConfig) -> list[str]:
    selected = OmegaConf.select(cfg, "stage0.selected_baselines")
    if isinstance(selected, ListConfig):
        return [str(item) for item in selected]
    if selected is None:
        return [str(key) for key in cfg.stage0.baselines.keys()]
    return [str(selected)]


def load_caption_map(cfg: DictConfig) -> dict[str, str]:
    configured = none_if_null(cfg.stage0.caption_metadata_path)
    if configured is None:
        return {}
    metadata_path = resolve_repo_path(configured)
    rows = read_metadata(metadata_path)
    captions: dict[str, str] = {}
    image_key = str(cfg.stage0.image_key)
    caption_key = str(cfg.stage0.caption_key)
    for row in rows:
        image_name = row.get(image_key)
        caption = row.get(caption_key)
        if isinstance(image_name, str) and isinstance(caption, str) and caption.strip():
            captions[image_name] = caption.strip()
    return captions


def discover_stage0_examples(cfg: DictConfig) -> list[Stage0Example]:
    input_dir = resolve_repo_path(str(cfg.stage0.input_dir))
    if not input_dir.exists():
        raise FileNotFoundError(f"Stage 0 input directory does not exist: {input_dir}")
    caption_map = load_caption_map(cfg)
    configured_limit = none_if_null(cfg.stage0.limit)
    limit = int(configured_limit) if configured_limit is not None else None
    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"No supported Stage 0 images found in {input_dir}")
    default_caption = str(cfg.stage0.default_caption)
    return [
        Stage0Example(
            image_path=image_path,
            caption=caption_map.get(image_path.name, default_caption),
        )
        for image_path in image_paths
    ]


def load_optional_vae(cfg: DictConfig, dtype: torch.dtype) -> AutoencoderKL | None:
    if not bool(cfg.models.vae.enabled):
        return None
    model_path = resolve_repo_path(str(cfg.models.vae.pretrained_model_name_or_path))
    if not model_path.exists():
        raise FileNotFoundError(f"Configured VAE is missing: {model_path}")
    return AutoencoderKL.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=bool(cfg.models.vae.local_files_only),
    )


def model_load_kwargs(model_cfg: DictConfig, dtype: torch.dtype) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": bool(model_cfg.local_files_only),
    }
    revision = none_if_null(model_cfg.get("revision"))
    variant = none_if_null(model_cfg.get("variant"))
    if revision is not None:
        kwargs["revision"] = revision
    if variant is not None:
        kwargs["variant"] = variant
    return kwargs


def control_runtime(cfg: DictConfig, control_name: str) -> ControlRuntime:
    control_cfg = cfg.models.controlnets[control_name]
    configured_precomputed = none_if_null(control_cfg.precomputed_dir)
    precomputed_dir = (
        resolve_repo_path(configured_precomputed) if configured_precomputed is not None else None
    )
    return ControlRuntime(
        name=control_name,
        scale=float(control_cfg.scale),
        start=float(control_cfg.start),
        end=float(control_cfg.end),
        preprocessor=str(control_cfg.preprocessor),
        precomputed_dir=precomputed_dir,
        threshold=int(control_cfg.threshold),
    )


def load_controlnet(cfg: DictConfig, control_name: str, dtype: torch.dtype) -> ControlNetModel:
    control_cfg = cfg.models.controlnets[control_name]
    model_path = resolve_repo_path(str(control_cfg.pretrained_model_name_or_path))
    if not model_path.exists():
        raise FileNotFoundError(f"ControlNet '{control_name}' is missing: {model_path}")
    return ControlNetModel.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=bool(control_cfg.local_files_only),
    )


def load_stage0_pipeline(
    cfg: DictConfig,
    baseline_cfg: DictConfig,
    dtype: torch.dtype,
) -> DiffusionPipeline:
    base_path = resolve_repo_path(str(cfg.models.base.pretrained_model_name_or_path))
    if not base_path.exists():
        raise FileNotFoundError(f"Configured SD1.5 base model is missing: {base_path}")
    control_names = [str(item) for item in baseline_cfg.controls]
    vae = load_optional_vae(cfg, dtype)
    load_kwargs = model_load_kwargs(cfg.models.base, dtype)
    if vae is not None:
        load_kwargs["vae"] = vae

    from_pretrained_kwargs = dict(load_kwargs)
    single_file_path = single_file_checkpoint_path(base_path)
    if control_names:
        controlnets = [load_controlnet(cfg, control_name, dtype) for control_name in control_names]
        from_pretrained_kwargs["controlnet"] = controlnets
        from_pretrained_kwargs["safety_checker"] = None
        if single_file_path is not None:
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                str(single_file_path),
                **from_pretrained_kwargs,
            )
        else:
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                str(base_path),
                **from_pretrained_kwargs,
            )
    else:
        from_pretrained_kwargs["safety_checker"] = None
        if single_file_path is not None:
            pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                str(single_file_path),
                **from_pretrained_kwargs,
            )
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                str(base_path),
                **from_pretrained_kwargs,
            )

    if bool(cfg.models.ip_adapter.enabled):
        ip_adapter_path = resolve_repo_path(
            str(cfg.models.ip_adapter.pretrained_model_name_or_path)
        )
        if not ip_adapter_path.exists():
            raise FileNotFoundError(f"IP-Adapter weights are missing: {ip_adapter_path}")
        pipe.load_ip_adapter(
            str(ip_adapter_path),
            subfolder=str(cfg.models.ip_adapter.subfolder),
            weight_name=str(cfg.models.ip_adapter.weight_name),
            local_files_only=bool(cfg.models.ip_adapter.local_files_only),
        )
        pipe.set_ip_adapter_scale(float(cfg.models.ip_adapter.scale))

    if bool(baseline_cfg.use_style_lora):
        lora_path = none_if_null(cfg.models.style_lora.path)
        if lora_path is None:
            raise ValueError("This baseline requires models.style_lora.path, but it is null.")
        resolved_lora_path = resolve_repo_path(lora_path)
        if not resolved_lora_path.exists():
            raise FileNotFoundError(f"Style LoRA is missing: {resolved_lora_path}")
        adapter_name = str(cfg.models.style_lora.adapter_name)
        pipe.load_lora_weights(
            str(resolved_lora_path),
            weight_name=str(cfg.models.style_lora.weight_name),
            adapter_name=adapter_name,
        )
        pipe.set_adapters(adapter_name, adapter_weights=float(cfg.models.style_lora.scale))

    if bool(cfg.stage0.attention_slicing):
        pipe.enable_attention_slicing()
    if bool(cfg.stage0.vae_slicing):
        pipe.enable_vae_slicing()
    if bool(cfg.stage0.model_cpu_offload):
        pipe.enable_model_cpu_offload()
    return pipe


def single_file_checkpoint_path(model_path: Path) -> Path | None:
    if model_path.is_file() and model_path.suffix == ".safetensors":
        return model_path
    if (model_path / "model_index.json").exists():
        return None
    checkpoint_paths = sorted(
        path
        for path in model_path.glob("*.safetensors")
        if "vae" not in path.name.lower() and "inpaint" not in path.name.lower()
    )
    if checkpoint_paths:
        return checkpoint_paths[0]
    return None


def threshold_edges(image: PILImage, threshold: int) -> PILImage:
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    lookup = [255 if value >= threshold else 0 for value in range(256)]
    binary = cast(PILImage, edges.point(lookup))
    return binary.convert("RGB")


def make_lineart(image: PILImage, threshold: int) -> PILImage:
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.CONTOUR)
    inverted = ImageOps.invert(edges)
    lookup = [0 if value < threshold else 255 for value in range(256)]
    binary = cast(PILImage, inverted.point(lookup))
    return binary.convert("RGB")


def load_precomputed_control(
    runtime: ControlRuntime,
    source_path: Path,
    width: int,
    height: int,
) -> PILImage:
    if runtime.precomputed_dir is None:
        raise ValueError(f"Control '{runtime.name}' requires precomputed_dir.")
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = runtime.precomputed_dir / f"{source_path.stem}{suffix}"
        if candidate.exists():
            return load_condition_image(candidate, width, height)
    raise FileNotFoundError(
        f"No precomputed {runtime.name} control image found for {source_path.name} "
        f"under {runtime.precomputed_dir}"
    )


def make_control_image(
    runtime: ControlRuntime,
    source_path: Path,
    width: int,
    height: int,
) -> PILImage:
    if runtime.preprocessor == "precomputed":
        return load_precomputed_control(runtime, source_path, width, height)
    source = load_condition_image(source_path, width, height)
    if runtime.preprocessor == "canny":
        return threshold_edges(source, runtime.threshold)
    if runtime.preprocessor == "lineart":
        return make_lineart(source, runtime.threshold)
    if runtime.preprocessor == "tile":
        return source
    raise ValueError(f"Unsupported control preprocessor: {runtime.preprocessor}")


def prompt_for_example(cfg: DictConfig, example: Stage0Example) -> str:
    template = str(cfg.stage0.prompt_template)
    return template.format(caption=example.caption)


def stage0_seed(cfg: DictConfig) -> int | None:
    configured_seed = none_if_null(cfg.stage0.seed)
    if configured_seed is None:
        return None
    return int(configured_seed)


def run_one_example(
    pipe: DiffusionPipeline,
    cfg: DictConfig,
    baseline_cfg: DictConfig,
    example: Stage0Example,
    device_name: str,
    index: int,
) -> PILImage:
    width = int(cfg.stage0.width)
    height = int(cfg.stage0.height)
    init_image = load_condition_image(example.image_path, width, height)
    generator = generators_for_batch(stage0_seed(cfg), device_name, index, 1)
    kwargs: dict[str, Any] = {
        "prompt": prompt_for_example(cfg, example),
        "negative_prompt": str(cfg.stage0.negative_prompt),
        "image": init_image,
        "strength": float(cfg.stage0.strength),
        "num_inference_steps": int(cfg.stage0.num_inference_steps),
        "guidance_scale": float(cfg.stage0.guidance_scale),
        "generator": generator,
        "height": height,
        "width": width,
    }
    if bool(cfg.models.ip_adapter.enabled):
        kwargs["ip_adapter_image"] = load_rgb_image(
            resolve_repo_path(str(cfg.stage0.style_image_path))
        )
    control_names = [str(item) for item in baseline_cfg.controls]
    if control_names:
        runtimes = [control_runtime(cfg, control_name) for control_name in control_names]
        kwargs["control_image"] = [
            make_control_image(runtime, example.image_path, width, height) for runtime in runtimes
        ]
        kwargs["controlnet_conditioning_scale"] = [runtime.scale for runtime in runtimes]
        kwargs["control_guidance_start"] = [runtime.start for runtime in runtimes]
        kwargs["control_guidance_end"] = [runtime.end for runtime in runtimes]
    result = cast(Any, pipe)(**kwargs)
    return cast(PILImage, cast(Any, result).images[0])


def run_baseline(cfg: DictConfig, baseline_key: str, examples: list[Stage0Example]) -> list[Path]:
    baseline_cfg = cfg.stage0.baselines[baseline_key]
    dtype = dtype_from_precision(str(cfg.stage0.mixed_precision))
    pipe = load_stage0_pipeline(cfg, baseline_cfg, dtype)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if not bool(cfg.stage0.model_cpu_offload):
        pipe.to(device_name)

    output_dir = resolve_repo_path(str(cfg.stage0.output_root)) / baseline_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for index, example in enumerate(examples):
        image = run_one_example(pipe, cfg, baseline_cfg, example, device_name, index)
        output_path = output_dir / f"{example.image_path.stem}-{index + 1:03d}.png"
        image.save(output_path)
        output_paths.append(output_path)
    return output_paths


def run(cfg: DictConfig) -> None:
    configure_environment(cfg)
    examples = discover_stage0_examples(cfg)
    report: dict[str, list[str]] = {}
    for baseline_key in selected_baseline_keys(cfg):
        report[baseline_key] = [str(path) for path in run_baseline(cfg, baseline_key, examples)]
    print(json.dumps(report, indent=2))


def main() -> None:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="sd15_style_control", overrides=sys.argv[1:])
    run(cfg)


__all__ = [
    "ControlRuntime",
    "Stage0Example",
    "configure_environment",
    "control_runtime",
    "discover_stage0_examples",
    "load_caption_map",
    "load_precomputed_control",
    "load_stage0_pipeline",
    "make_control_image",
    "make_lineart",
    "prompt_for_example",
    "run",
    "run_baseline",
    "run_one_example",
    "selected_baseline_keys",
    "stage0_seed",
    "threshold_edges",
]
