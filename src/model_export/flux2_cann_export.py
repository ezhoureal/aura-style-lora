from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from omegaconf import DictConfig
from torch import Tensor, nn

from model_export.cann_export import CannModelSpec, relative_path

from lora.local_edit_common import (
    dtype_from_precision,
    none_if_null,
    resolve_repo_path,
)


ONNX_INPUT_NAMES = [
    "hidden_states",
    "timestep",
    "guidance",
    "encoder_hidden_states",
    "txt_ids",
    "img_ids",
]
ONNX_OUTPUT_NAMES = ["sample"]


@dataclass(frozen=True)
class Flux2DenoiserShape:
    batch_size: int
    height: int
    width: int
    max_sequence_length: int
    prompt_embed_dim: int
    packed_latent_channels: int

    @property
    def packed_latent_tokens(self) -> int:
        return (self.height // 16) * (self.width // 16)

    @property
    def denoiser_tokens(self) -> int:
        return self.packed_latent_tokens * 2

    @property
    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "hidden_states": (
                self.batch_size,
                self.denoiser_tokens,
                self.packed_latent_channels,
            ),
            "timestep": (self.batch_size,),
            "guidance": (self.batch_size,),
            "encoder_hidden_states": (
                self.batch_size,
                self.max_sequence_length,
                self.prompt_embed_dim,
            ),
            "txt_ids": (self.batch_size, self.max_sequence_length, 4),
            "img_ids": (self.batch_size, self.denoiser_tokens, 4),
        }


def shape_from_config(cfg: DictConfig, transformer: nn.Module) -> Flux2DenoiserShape:
    model_cfg = cfg.models
    transformer_config = cast(Any, transformer).config
    shape = Flux2DenoiserShape(
        batch_size=int(model_cfg.batch_size),
        height=int(model_cfg.height),
        width=int(model_cfg.width),
        max_sequence_length=int(model_cfg.max_sequence_length),
        prompt_embed_dim=int(transformer_config.joint_attention_dim),
        packed_latent_channels=int(transformer_config.in_channels),
    )
    validate_shape(shape)
    return shape


@dataclass(frozen=True)
class Flux2DenoiserInputs:
    hidden_states: Tensor
    timestep: Tensor
    guidance: Tensor
    encoder_hidden_states: Tensor
    txt_ids: Tensor
    img_ids: Tensor

    def as_tuple(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.hidden_states,
            self.timestep,
            self.guidance,
            self.encoder_hidden_states,
            self.txt_ids,
            self.img_ids,
        )


class Flux2DenoiserExportWrapper(nn.Module):
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        guidance: Tensor,
        encoder_hidden_states: Tensor,
        txt_ids: Tensor,
        img_ids: Tensor,
    ) -> Tensor:
        result = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )
        return cast(Tensor, result[0])


class ExportableRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: tuple[int, ...],
        eps: float,
        weight: Tensor | None,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        if weight is None:
            self.register_parameter("weight", None)
        else:
            self.weight = nn.Parameter(weight.detach().clone())

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance_dims = tuple(range(-len(self.normalized_shape), 0))
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=variance_dims, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        if self.weight is None:
            return normalized
        return normalized * self.weight

    @classmethod
    def from_torch_rms_norm(cls, module: nn.RMSNorm) -> ExportableRMSNorm:
        normalized_shape = cast(
            tuple[int, ...], tuple(int(value) for value in module.normalized_shape)
        )
        weight = cast(Tensor | None, module.weight)
        eps = torch.finfo(torch.float32).eps if module.eps is None else float(module.eps)
        return cls(normalized_shape, eps, weight)


def replace_rms_norm_modules(module: nn.Module) -> None:
    for child_name, child in module.named_children():
        if isinstance(child, nn.RMSNorm):
            module.add_module(child_name, ExportableRMSNorm.from_torch_rms_norm(child))
        else:
            replace_rms_norm_modules(child)


def validate_shape(shape: Flux2DenoiserShape) -> None:
    if shape.batch_size < 1:
        raise ValueError("models.batch_size must be at least 1")
    if shape.height % 16 != 0 or shape.width % 16 != 0:
        raise ValueError("models.height and models.width must be divisible by 16")
    if shape.max_sequence_length < 1:
        raise ValueError("models.max_sequence_length must be at least 1")
    if shape.prompt_embed_dim < 1:
        raise ValueError("transformer.config.joint_attention_dim must be at least 1")
    if shape.packed_latent_channels < 1:
        raise ValueError("transformer.config.in_channels must be at least 1")


def dummy_inputs(
    shape: Flux2DenoiserShape, dtype: torch.dtype, device: torch.device
) -> Flux2DenoiserInputs:
    return Flux2DenoiserInputs(
        hidden_states=torch.zeros(
            shape.batch_size,
            shape.denoiser_tokens,
            shape.packed_latent_channels,
            device=device,
            dtype=dtype,
        ),
        timestep=torch.full((shape.batch_size,), 0.5, device=device, dtype=dtype),
        guidance=torch.full((shape.batch_size,), 4.0, device=device, dtype=torch.float32),
        encoder_hidden_states=torch.zeros(
            shape.batch_size,
            shape.max_sequence_length,
            shape.prompt_embed_dim,
            device=device,
            dtype=dtype,
        ),
        txt_ids=torch.zeros(
            shape.batch_size,
            shape.max_sequence_length,
            4,
            device=device,
            dtype=torch.float32,
        ),
        img_ids=torch.zeros(
            shape.batch_size,
            shape.denoiser_tokens,
            4,
            device=device,
            dtype=torch.float32,
        ),
    )


def checkpoint_dir_from_config(cfg: DictConfig) -> Path:
    configured = none_if_null(cfg.models.checkpoint_dir)
    if configured is not None:
        return resolve_repo_path(configured)
    raise ValueError(
        "models.checkpoint_dir must be set to the LoRA checkpoint directory for CANN export."
    )


def load_fused_transformer(cfg: DictConfig, checkpoint_dir: Path) -> nn.Module:
    model_cfg = cfg.models
    dtype = dtype_from_precision(str(model_cfg.precision))
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": bool(model_cfg.get("local_files_only", True)),
    }
    revision = none_if_null(model_cfg.get("revision"))
    variant = none_if_null(model_cfg.get("variant"))
    if revision is not None:
        load_kwargs["revision"] = revision
    if variant is not None:
        load_kwargs["variant"] = variant

    pipe = DiffusionPipeline.from_pretrained(
        str(model_cfg.pretrained_model_name_or_path),
        **load_kwargs,
    )
    transformer = cast(Any, pipe.transformer)
    transformer.load_lora_adapter(
        str(checkpoint_dir),
        prefix=None,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="aura",
    )
    transformer.fuse_lora(
        lora_scale=float(model_cfg.lora_scale),
        safe_fusing=bool(model_cfg.safe_fusing),
        adapter_names=["aura"],
    )
    transformer.unload_lora()
    transformer_module = cast(nn.Module, transformer)
    replace_rms_norm_modules(transformer_module)
    return transformer_module


def flux2_model_spec(
    shape: Flux2DenoiserShape,
    wrapper: Flux2DenoiserExportWrapper | None = None,
    inputs: Flux2DenoiserInputs | None = None,
    metadata: dict[str, Any] | None = None,
) -> CannModelSpec:
    return CannModelSpec(
        input_names=tuple(ONNX_INPUT_NAMES),
        output_names=tuple(ONNX_OUTPUT_NAMES),
        input_shapes=shape.input_shapes,
        metadata={} if metadata is None else metadata,
        model=wrapper,
        inputs=() if inputs is None else inputs.as_tuple(),
    )


def flux2_manifest_metadata(
    cfg: DictConfig, checkpoint_dir: Path, shape: Flux2DenoiserShape
) -> dict[str, Any]:
    return {
        "model_key": str(cfg.models.name),
        "checkpoint_dir": relative_path(checkpoint_dir),
        "shape": asdict(shape),
        "lora_fused": True,
    }


def build_flux2_model_spec(cfg: DictConfig, device: torch.device) -> CannModelSpec:
    checkpoint_dir = checkpoint_dir_from_config(cfg)
    dtype = dtype_from_precision(str(cfg.models.precision))
    transformer = load_fused_transformer(cfg, checkpoint_dir).to(device)
    shape = shape_from_config(cfg, transformer)
    wrapper = Flux2DenoiserExportWrapper(transformer)
    return flux2_model_spec(
        shape,
        wrapper,
        dummy_inputs(shape, dtype, device),
        flux2_manifest_metadata(cfg, checkpoint_dir, shape),
    )
