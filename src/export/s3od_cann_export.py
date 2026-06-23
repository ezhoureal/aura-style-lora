from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from .cann_export import CannModelSpec
from lora.local_edit_common import dtype_from_precision


class S3ODExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        outputs = cast(dict[str, Tensor], self.model(pixel_values))
        return outputs["pred_masks"], outputs["pred_iou"]


def load_s3od_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    s3od = import_module("s3od")
    detector = cast(Any, s3od).BackgroundRemoval(
        model_id=str(cfg.models.pretrained_model_name_or_path),
        image_size=int(cfg.models.image_size),
        device=str(device),
    )
    dtype = dtype_from_precision(str(cfg.models.precision))
    return cast(nn.Module, detector.model).to(device=device, dtype=dtype)


def s3od_model_spec(
    cfg: DictConfig,
    model: nn.Module | None = None,
    device: torch.device | None = None,
) -> CannModelSpec:
    batch_size = int(cfg.models.batch_size)
    image_size = int(cfg.models.image_size)
    inputs: tuple[Tensor, ...] = ()
    wrapper: nn.Module | None = None
    if model is not None:
        if device is None:
            raise ValueError("device is required when constructing an exportable S3OD spec")
        dtype = dtype_from_precision(str(cfg.models.precision))
        wrapper = S3ODExportWrapper(model)
        inputs = (
            torch.zeros(
                batch_size,
                3,
                image_size,
                image_size,
                dtype=dtype,
                device=device,
            ),
        )
    return CannModelSpec(
        input_names=("pixel_values",),
        output_names=("pred_masks", "pred_iou"),
        input_shapes={"pixel_values": (batch_size, 3, image_size, image_size)},
        metadata={
            "model_key": str(cfg.models.name),
            "pretrained_model_name_or_path": str(cfg.models.pretrained_model_name_or_path),
            "image_size": image_size,
            "input_preprocessing": {
                "color_space": "RGB",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize": "letterbox",
            },
        },
        model=wrapper,
        inputs=inputs,
    )


def build_s3od_model_spec(cfg: DictConfig, device: torch.device) -> CannModelSpec:
    return s3od_model_spec(cfg, load_s3od_model(cfg, device), device)
