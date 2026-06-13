from __future__ import annotations

import json
import sys
from collections.abc import Callable
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
from lora.local_edit_sd3 import (
    SD3PairedEditDataset,
    SD3PreparedBatch,
    StableDiffusion3PairedEditLoraTrainer,
    apply_sd3_input_projection_patch,
    collate_sd3_batches,
    expand_sd3_transformer_input_for_paired_edit,
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


TrainerFactory = Callable[[DictConfig, str], EditTrainer]

TRAINER_FACTORIES: dict[str, TrainerFactory] = {
    "stable_diffusion_ip2p_lora": StableDiffusionIp2PLoraTrainer,
    "stable_diffusion_3_paired_edit_lora": StableDiffusion3PairedEditLoraTrainer,
    "flux2_paired_edit_lora": Flux2PairedEditLoraTrainer,
    "unsupported_local_edit": UnsupportedLocalEditTrainer,
}


def make_trainer(cfg: DictConfig, model_key: str) -> EditTrainer:
    trainer_name = str(cfg.models[model_key].trainer)
    trainer_factory = TRAINER_FACTORIES.get(trainer_name)
    if trainer_factory is not None:
        return trainer_factory(cfg, model_key)
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
    "SD3PairedEditDataset",
    "SD3PreparedBatch",
    "StableDiffusionIp2PLoraTrainer",
    "StableDiffusion3PairedEditLoraTrainer",
    "TRAINER_FACTORIES",
    "TrainerFactory",
    "UnsupportedLocalEditTrainer",
    "apply_lora_checkpoint",
    "apply_sd3_input_projection_patch",
    "checkpoint_step_from_path",
    "collate_batches",
    "collate_flux_batches",
    "collate_sd3_batches",
    "configure_environment",
    "dtype_from_precision",
    "expand_unet_conv_in_for_ip2p",
    "expand_sd3_transformer_input_for_paired_edit",
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
