from __future__ import annotations

import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from lora.local_edit_common import (
    REPO_ROOT,
    SUPPORTED_IMAGE_SUFFIXES,
    EditTrainer,
    PairExample,
    ResumeCheckpoint,
    apply_lora_checkpoint,
    checkpoint_step_from_path,
    configure_environment,
    dtype_from_precision,
    freeze_module,
    latest_checkpoint,
    load_pair_examples,
    load_rgb_image,
    make_training_progress,
    none_if_null,
    read_metadata,
    resolve_relative_path,
    resolve_resume_checkpoint,
    resolve_repo_path,
    selected_model_keys,
    trainable_parameters,
)
from lora.local_edit_flux2 import (
    Flux2PairedEditLoraTrainer,
    FluxPairedEditDataset,
    FluxPreparedBatch,
    collate_flux_batches,
    flow_match_noisy_latents,
    flow_match_training_target,
)
from lora.local_edit_sd import (
    PairedEditDataset,
    PreparedBatch,
    StableDiffusionIp2PLoraTrainer,
    collate_batches,
    expand_unet_conv_in_for_ip2p,
)


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
    if trainer_name == "flux2_paired_edit_lora":
        return Flux2PairedEditLoraTrainer(cfg, model_key)
    if trainer_name == "unsupported_local_edit":
        return UnsupportedLocalEditTrainer(cfg, model_key)
    raise ValueError(f"Unknown trainer adapter: {trainer_name}")


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


__all__ = [
    "REPO_ROOT",
    "SUPPORTED_IMAGE_SUFFIXES",
    "EditTrainer",
    "Flux2PairedEditLoraTrainer",
    "FluxPairedEditDataset",
    "FluxPreparedBatch",
    "PairExample",
    "PairedEditDataset",
    "PreparedBatch",
    "ResumeCheckpoint",
    "StableDiffusionIp2PLoraTrainer",
    "UnsupportedLocalEditTrainer",
    "apply_lora_checkpoint",
    "checkpoint_step_from_path",
    "collate_batches",
    "collate_flux_batches",
    "configure_environment",
    "dtype_from_precision",
    "expand_unet_conv_in_for_ip2p",
    "flow_match_noisy_latents",
    "flow_match_training_target",
    "freeze_module",
    "latest_checkpoint",
    "load_pair_examples",
    "load_rgb_image",
    "make_training_progress",
    "make_trainer",
    "none_if_null",
    "read_metadata",
    "resolve_relative_path",
    "resolve_resume_checkpoint",
    "resolve_repo_path",
    "run",
    "selected_model_keys",
    "trainable_parameters",
]
