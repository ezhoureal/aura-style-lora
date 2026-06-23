from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from model_export.cann_export import CannModelSpec, run_export_pipeline
from model_export.flux2_cann_export import build_flux2_model_spec
from model_export.s3od_cann_export import build_s3od_model_spec


def build_model_spec(cfg: DictConfig, device: torch.device) -> CannModelSpec:
    kind = str(cfg.models.kind)
    if kind == "flux2":
        return build_flux2_model_spec(cfg, device)
    if kind == "s3od":
        return build_s3od_model_spec(cfg, device)
    raise ValueError(f"Unsupported export model kind: {kind}")


@hydra.main(
    version_base=None,
    config_path=str("configs/export"),
    config_name="export",
)
def main(cfg: DictConfig) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return run_export_pipeline(
        cfg, Path(str(cfg.export.output_root)), build_model_spec(cfg, device)
    )


if __name__ == "__main__":
    main()
