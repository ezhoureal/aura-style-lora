import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch import Tensor, nn

from lora.local_edit_common import REPO_ROOT
from export.s3od_cann_export import s3od_model_spec


class TinyS3OD(nn.Module):
    def forward(self, pixel_values: Tensor) -> dict[str, Tensor]:
        batch_size, _, height, width = pixel_values.shape
        return {
            "pred_masks": torch.zeros(batch_size, 3, height, width),
            "pred_iou": torch.zeros(batch_size, 3),
            "features": torch.zeros(batch_size, 1),
        }


def compose_export_config(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs" / "export"), version_base=None):
        return compose(config_name="export", overrides=list(overrides))


def test_hierarchical_export_config_selects_s3od() -> None:
    cfg = compose_export_config("models=s3od-dis")

    assert cfg.models.kind == "s3od"
    assert cfg.export.output_root == "outputs/export/s3od-dis"
    assert cfg.export.cann.output_name == "s3od_dis"


def test_s3od_model_spec_exports_masks_and_iou() -> None:
    cfg = compose_export_config("models=s3od-dis", "models.image_size=64")
    spec = s3od_model_spec(cfg, TinyS3OD(), torch.device("cpu"))

    assert spec.input_names == ("pixel_values",)
    assert spec.output_names == ("pred_masks", "pred_iou")
    assert spec.input_shapes == {"pixel_values": (1, 3, 64, 64)}
    assert spec.inputs[0].shape == (1, 3, 64, 64)
    assert spec.model is not None
    pred_masks, pred_iou = spec.model(*spec.inputs)
    assert pred_masks.shape == (1, 3, 64, 64)
    assert pred_iou.shape == (1, 3)
