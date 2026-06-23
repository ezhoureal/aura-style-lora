from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from lora.local_edit_common import REPO_ROOT, configure_environment, resolve_repo_path

from .cann_export import CannModelSpec, run_export_pipeline
from .flux2_cann_export import build_flux2_model_spec
from .s3od_cann_export import build_s3od_model_spec


def build_model_spec(cfg: DictConfig, device: torch.device) -> CannModelSpec:
    kind = str(cfg.models.kind)
    if kind == "flux2":
        return build_flux2_model_spec(cfg, device)
    if kind == "s3od":
        return build_s3od_model_spec(cfg, device)
    raise ValueError(f"Unsupported export model kind: {kind}")


@hydra.main(
    version_base=None,
    config_path=str(REPO_ROOT / "configs" / "export"),
    config_name="export",
)
def main(cfg: DictConfig) -> Path:
    configure_environment(cfg)
    output_dir = resolve_repo_path(str(cfg.export.output_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return run_export_pipeline(cfg, output_dir, build_model_spec(cfg, device))
