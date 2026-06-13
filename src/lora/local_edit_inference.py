from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
)
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL.Image import Image as PILImage

from lora.local_edit_common import (
    REPO_ROOT,
    batched_paths,
    configure_environment,
    discover_images,
    evaluation_seed,
    generators_for_batch,
    load_rgb_image,
    none_if_null,
    resolve_repo_path,
    selected_model_keys,
)
from lora.local_edit_flux2 import load_flux2_pipeline, run_flux2_batch
from lora.local_edit_sd import load_sd_pipeline, run_sd_batch
from lora.local_edit_sd3 import load_sd3_pipeline, run_sd3_batch


PipelineLoader = Callable[[DictConfig, str, Path], Any]
BatchRunner = Callable[[Any, DictConfig, list[Path], str, int], list[PILImage]]


def run_sd_batch_adapter(
    pipe: Any,
    cfg: DictConfig,
    input_paths: list[Path],
    device_name: str,
    batch_offset: int,
) -> list[PILImage]:
    return run_sd_batch(
        cast(StableDiffusionInstructPix2PixPipeline, pipe),
        cfg,
        input_paths,
        device_name,
        batch_offset,
    )


def run_diffusion_batch_adapter(
    runner: Callable[[DiffusionPipeline, DictConfig, list[Path], str, int], list[PILImage]],
    pipe: Any,
    cfg: DictConfig,
    input_paths: list[Path],
    device_name: str,
    batch_offset: int,
) -> list[PILImage]:
    return runner(cast(DiffusionPipeline, pipe), cfg, input_paths, device_name, batch_offset)


def run_sd3_batch_adapter(
    pipe: Any,
    cfg: DictConfig,
    input_paths: list[Path],
    device_name: str,
    batch_offset: int,
) -> list[PILImage]:
    return run_diffusion_batch_adapter(
        run_sd3_batch, pipe, cfg, input_paths, device_name, batch_offset
    )


def run_flux2_batch_adapter(
    pipe: Any,
    cfg: DictConfig,
    input_paths: list[Path],
    device_name: str,
    batch_offset: int,
) -> list[PILImage]:
    return run_diffusion_batch_adapter(
        run_flux2_batch,
        pipe,
        cfg,
        input_paths,
        device_name,
        batch_offset,
    )


PIPELINE_LOADERS: dict[str, PipelineLoader] = {
    "stable_diffusion_ip2p_lora": load_sd_pipeline,
    "stable_diffusion_3_paired_edit_lora": load_sd3_pipeline,
    "flux2_paired_edit_lora": load_flux2_pipeline,
}

BATCH_RUNNERS: dict[str, BatchRunner] = {
    "stable_diffusion_ip2p_lora": run_sd_batch_adapter,
    "stable_diffusion_3_paired_edit_lora": run_sd3_batch_adapter,
    "flux2_paired_edit_lora": run_flux2_batch_adapter,
}


def checkpoint_dir_for_model(cfg: DictConfig, model_key: str) -> Path:
    configured = cfg.evaluation.checkpoint_dir
    if configured is not None:
        return resolve_repo_path(str(configured))
    output_root = resolve_repo_path(str(cfg.training.output_root))
    return output_root / model_key / f"checkpoint-{int(cfg.training.max_train_steps):06d}"


def load_pipeline(cfg: DictConfig, model_key: str, checkpoint_dir: Path) -> Any:
    model_cfg = cfg.models[model_key]
    trainer = str(model_cfg.trainer)
    pipeline_loader = PIPELINE_LOADERS.get(trainer)
    if pipeline_loader is not None:
        return pipeline_loader(cfg, model_key, checkpoint_dir)
    raise NotImplementedError(f"Local inference is not implemented for trainer {trainer}")


def run_batch(
    pipe: Any,
    cfg: DictConfig,
    model_key: str,
    input_paths: list[Path],
    device_name: str,
    batch_offset: int,
) -> list[PILImage]:
    trainer = str(cfg.models[model_key].trainer)
    batch_runner = BATCH_RUNNERS.get(trainer)
    if batch_runner is not None:
        return batch_runner(pipe, cfg, input_paths, device_name, batch_offset)
    raise NotImplementedError(f"Local inference is not implemented for trainer {trainer}")


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
    pipe = load_pipeline(cfg, model_key, checkpoint_dir)
    pipe.to(device_name)
    output_dir = resolve_repo_path(str(cfg.evaluation.output_root)) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    batch_size = int(cfg.evaluation.batch_size)
    for batch_index, batch_paths in enumerate(batched_paths(image_paths, batch_size)):
        batch_offset = batch_index * batch_size
        images = run_batch(
            pipe=pipe,
            cfg=cfg,
            model_key=model_key,
            input_paths=batch_paths,
            device_name=device_name,
            batch_offset=batch_offset,
        )
        if len(images) != len(batch_paths):
            raise RuntimeError(
                f"Expected {len(batch_paths)} outputs from pipeline, received {len(images)}."
            )
        for offset, image in enumerate(images):
            input_path = batch_paths[offset]
            output_path = output_dir / f"{input_path.stem}-{batch_offset + offset + 1:03d}.png"
            image.save(output_path)
            output_paths.append(output_path)
    return output_paths


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


__all__ = [
    "batched_paths",
    "BatchRunner",
    "checkpoint_dir_for_model",
    "configure_environment",
    "discover_images",
    "evaluation_seed",
    "generators_for_batch",
    "load_flux2_pipeline",
    "load_pipeline",
    "load_rgb_image",
    "PipelineLoader",
    "PIPELINE_LOADERS",
    "BATCH_RUNNERS",
    "load_sd3_pipeline",
    "load_sd_pipeline",
    "run",
    "run_batch",
    "run_flux2_batch_adapter",
    "run_flux2_batch",
    "run_model",
    "run_sd3_batch_adapter",
    "run_sd3_batch",
    "run_sd_batch_adapter",
    "run_sd_batch",
]
