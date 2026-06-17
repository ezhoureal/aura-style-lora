from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import onnx
from onnx import TensorProto, numpy_helper
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from lora.local_edit_common import (
    REPO_ROOT,
    configure_environment,
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
SUPPORTED_QUANTIZATION_MODES = {"none", "int8_dynamic", "int4_palette"}
SUPPORTED_CANN_TARGETS = {"om", "omc", "tiny", "ispnn", "security"}
SANITIZER_EXTERNAL_DATA_MARKER = "lora_flux_omg_sanitized"
SANITIZER_VERSION = "7"
MAX_DUPLICATED_INITIALIZER_BYTES = 1024


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


def shape_with_transformer_config(
    shape: Flux2DenoiserShape,
    transformer: nn.Module,
) -> Flux2DenoiserShape:
    transformer_config = cast(Any, transformer).config
    resolved_shape = Flux2DenoiserShape(
        batch_size=shape.batch_size,
        height=shape.height,
        width=shape.width,
        max_sequence_length=shape.max_sequence_length,
        prompt_embed_dim=int(transformer_config.joint_attention_dim),
        packed_latent_channels=int(transformer_config.in_channels),
    )
    validate_shape(resolved_shape)
    return resolved_shape


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
        normalized = hidden_states * torch.rsqrt(variance.to(hidden_states.dtype) + self.eps)
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


def shape_from_config(cfg: DictConfig) -> Flux2DenoiserShape:
    export_cfg = cfg.export
    shape = Flux2DenoiserShape(
        batch_size=int(export_cfg.batch_size),
        height=int(export_cfg.height),
        width=int(export_cfg.width),
        max_sequence_length=int(export_cfg.max_sequence_length),
        prompt_embed_dim=int(export_cfg.prompt_embed_dim),
        packed_latent_channels=int(export_cfg.packed_latent_channels),
    )
    validate_shape(shape)
    return shape


def validate_shape(shape: Flux2DenoiserShape) -> None:
    if shape.batch_size < 1:
        raise ValueError("export.batch_size must be at least 1")
    if shape.height % 16 != 0 or shape.width % 16 != 0:
        raise ValueError("export.height and export.width must be divisible by 16")
    if shape.max_sequence_length < 1:
        raise ValueError("export.max_sequence_length must be at least 1")
    if shape.prompt_embed_dim < 1:
        raise ValueError("export.prompt_embed_dim must be at least 1")
    if shape.packed_latent_channels < 1:
        raise ValueError("export.packed_latent_channels must be at least 1")


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
    configured = none_if_null(cfg.export.checkpoint_dir)
    if configured is not None:
        return resolve_repo_path(configured)
    raise ValueError(
        "export.checkpoint_dir must be set to the LoRA checkpoint directory for CANN export."
    )


def load_fused_transformer(cfg: DictConfig, checkpoint_dir: Path) -> nn.Module:
    model_cfg = cfg.models[str(cfg.model_key)]
    dtype = dtype_from_precision(str(cfg.export.precision))
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
    transformer.set_adapters(["aura"], weights=[float(cfg.export.lora_scale)])
    transformer.fuse_lora(
        lora_scale=float(cfg.export.lora_scale),
        safe_fusing=bool(cfg.export.safe_fusing),
        adapter_names=["aura"],
    )
    transformer.unload_lora()
    transformer_module = cast(nn.Module, transformer)
    replace_rms_norm_modules(transformer_module)
    return transformer_module


def export_onnx(
    wrapper: Flux2DenoiserExportWrapper,
    inputs: Flux2DenoiserInputs,
    output_path: Path,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs.as_tuple(),
            output_path,
            input_names=ONNX_INPUT_NAMES,
            output_names=ONNX_OUTPUT_NAMES,
            opset_version=opset,
            do_constant_folding=True,
        )


def quantized_output_path(cfg: DictConfig, output_dir: Path) -> Path:
    return output_dir / str(cfg.export.quantized_onnx_filename)


def remove_external_data_file(onnx_path: Path) -> None:
    external_data_path = onnx_path.with_name(f"{onnx_path.name}.data")
    if external_data_path.exists():
        external_data_path.unlink()


def strip_intermediate_value_info(onnx_path: Path, output_dir: Path) -> Path:
    stripped_path = output_dir / f"{onnx_path.stem}.shape_stripped{onnx_path.suffix}"
    model = onnx.load(onnx_path, load_external_data=True)
    del model.graph.value_info[:]
    remove_external_data_file(stripped_path)
    onnx.save_model(
        model,
        stripped_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{stripped_path.name}.data",
        size_threshold=1024,
    )
    return stripped_path


def model_is_marked_sanitized(model: onnx.ModelProto) -> bool:
    return any(
        prop.key == SANITIZER_EXTERNAL_DATA_MARKER and prop.value == SANITIZER_VERSION
        for prop in model.metadata_props
    )


def mark_model_sanitized(model: onnx.ModelProto) -> None:
    for prop in model.metadata_props:
        if prop.key == SANITIZER_EXTERNAL_DATA_MARKER:
            prop.value = SANITIZER_VERSION
            return
    prop = model.metadata_props.add()
    prop.key = SANITIZER_EXTERNAL_DATA_MARKER
    prop.value = SANITIZER_VERSION


def transpose_perm(node: onnx.NodeProto) -> list[int]:
    for attribute in node.attribute:
        if attribute.name == "perm":
            return [int(value) for value in attribute.ints]
    return []


def rewrite_rank3_transpose_for_omg(model: onnx.ModelProto) -> None:
    rewritten_nodes: list[onnx.NodeProto] = []
    existing_initializer_names = {initializer.name for initializer in model.graph.initializer}
    shapes_by_name: dict[str, list[int]] = {}
    for value_info in (
        list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    ):
        shape = value_info.type.tensor_type.shape
        if all(dim.HasField("dim_value") for dim in shape.dim):
            shapes_by_name[value_info.name] = [int(dim.dim_value) for dim in shape.dim]

    for node in model.graph.node:
        if node.op_type != "Transpose" or transpose_perm(node) != [0, 2, 1]:
            rewritten_nodes.append(node)
            continue
        output_shape = shapes_by_name.get(node.output[0])
        if output_shape is None:
            rewritten_nodes.append(node)
            continue

        axes_name = f"{node.output[0]}_omg_rank4_axes"
        if axes_name not in existing_initializer_names:
            axes = numpy_helper.from_array(torch.tensor([1], dtype=torch.int64).numpy(), axes_name)
            model.graph.initializer.append(axes)
            existing_initializer_names.add(axes_name)

        unsqueeze_output = f"{node.output[0]}_omg_rank4_unsqueeze"
        transpose_output = f"{node.output[0]}_omg_rank4_transpose"
        shape_name = f"{node.output[0]}_omg_rank3_shape"
        if shape_name not in existing_initializer_names:
            shape_tensor = numpy_helper.from_array(
                torch.tensor(output_shape, dtype=torch.int64).numpy(),
                shape_name,
            )
            model.graph.initializer.append(shape_tensor)
            existing_initializer_names.add(shape_name)
        rewritten_nodes.extend(
            [
                onnx.helper.make_node(
                    "Unsqueeze",
                    [node.input[0], axes_name],
                    [unsqueeze_output],
                    name=f"{node.name}_omg_rank4_unsqueeze",
                ),
                onnx.helper.make_node(
                    "Transpose",
                    [unsqueeze_output],
                    [transpose_output],
                    name=f"{node.name}_omg_rank4_transpose",
                    perm=[0, 1, 3, 2],
                ),
                onnx.helper.make_node(
                    "Reshape",
                    [transpose_output, shape_name],
                    [node.output[0]],
                    name=f"{node.name}_omg_rank3_reshape",
                ),
            ]
        )

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)


def expand_clip_scalar_bounds_for_omg(model: onnx.ModelProto) -> None:
    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    clip_bound_names = {
        input_name
        for node in model.graph.node
        if node.op_type == "Clip"
        for input_name in node.input[1:]
        if input_name in initializer_by_name
    }
    for initializer_name in clip_bound_names:
        initializer = initializer_by_name[initializer_name]
        if len(initializer.dims) != 0:
            continue
        value = numpy_helper.to_array(initializer).reshape(1)
        initializer.CopyFrom(numpy_helper.from_array(value, name=initializer.name))


def rewrite_clip_for_omg(model: onnx.ModelProto) -> None:
    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if (
            node.op_type != "Clip"
            or len(node.input) != 3
            or node.input[1] == ""
            or node.input[2] == ""
        ):
            rewritten_nodes.append(node)
            continue
        max_output = f"{node.output[0]}_omg_clip_min"
        rewritten_nodes.extend(
            [
                onnx.helper.make_node(
                    "Max",
                    [node.input[0], node.input[1]],
                    [max_output],
                    name=f"{node.name}_omg_clip_min",
                ),
                onnx.helper.make_node(
                    "Min",
                    [max_output, node.input[2]],
                    list(node.output),
                    name=f"{node.name}_omg_clip_max",
                ),
            ]
        )

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)


def static_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for value_info in (
        list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    ):
        dims = value_info.type.tensor_type.shape.dim
        if all(dim.HasField("dim_value") for dim in dims):
            shapes[value_info.name] = [int(dim.dim_value) for dim in dims]
    return shapes


def rewrite_static_squeeze_ops_for_omg(model: onnx.ModelProto) -> None:
    shapes_by_name = static_shapes(model)
    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        output_shape = shapes_by_name.get(node.output[0]) if len(node.output) == 1 else None
        if node.op_type not in {"Squeeze", "Unsqueeze"} or output_shape is None:
            rewritten_nodes.append(node)
            continue
        shape_name = f"{node.output[0]}_omg_static_shape"
        model.graph.initializer.append(
            numpy_helper.from_array(
                torch.tensor(output_shape, dtype=torch.int64).numpy(), shape_name
            )
        )
        rewritten_nodes.append(
            onnx.helper.make_node(
                "Reshape",
                [node.input[0], shape_name],
                list(node.output),
                name=f"{node.name}_omg_static_reshape",
            )
        )

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)


def initializer_byte_size(initializer: onnx.TensorProto) -> int:
    if len(initializer.raw_data) > 0:
        return len(initializer.raw_data)
    return int(numpy_helper.to_array(initializer).nbytes)


def duplicate_shared_small_initializers_for_omg(model: onnx.ModelProto) -> None:
    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    uses_by_initializer: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in model.graph.node:
        for input_index, input_name in enumerate(node.input):
            if input_name in initializer_by_name:
                uses_by_initializer[input_name].append((node, input_index))

    existing_names = set(initializer_by_name)
    for initializer_name, uses in uses_by_initializer.items():
        if len(uses) < 2:
            continue
        initializer = initializer_by_name[initializer_name]
        if initializer_byte_size(initializer) > MAX_DUPLICATED_INITIALIZER_BYTES:
            continue
        for duplicate_index, (node, input_index) in enumerate(uses[1:], start=1):
            duplicate_name = f"{initializer_name}_omg_const_{duplicate_index}"
            if duplicate_name in existing_names:
                raise ValueError(f"Duplicate initializer name already exists: {duplicate_name}")
            duplicate = onnx.TensorProto()
            duplicate.CopyFrom(initializer)
            duplicate.name = duplicate_name
            model.graph.initializer.append(duplicate)
            node.input[input_index] = duplicate_name
            existing_names.add(duplicate_name)


def rewrite_double_casts_for_omg(model: onnx.ModelProto) -> None:
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == TensorProto.DOUBLE:
                attribute.i = TensorProto.FLOAT

    for value_info in list(model.graph.value_info) + list(model.graph.output):
        if value_info.type.tensor_type.elem_type == TensorProto.DOUBLE:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT


def sanitize_onnx_for_omg(onnx_path: Path) -> None:
    model = onnx.load(onnx_path, load_external_data=True)
    if model_is_marked_sanitized(model):
        return
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for initializer in model.graph.initializer:
        if initializer.data_type != TensorProto.DOUBLE:
            continue
        tensor = numpy_helper.from_array(
            numpy_helper.to_array(initializer).astype("float32"),
            name=initializer.name,
        )
        initializer.CopyFrom(tensor)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name != "value" or attribute.t.data_type != TensorProto.DOUBLE:
                    continue
                tensor = numpy_helper.from_array(
                    numpy_helper.to_array(attribute.t).astype("float32"),
                    name=attribute.t.name,
                )
                attribute.t.CopyFrom(tensor)
        if node.op_type == "LayerNormalization" and len(node.input) == 2:
            scale_name = node.input[1]
            scale_initializer = next(
                initializer
                for initializer in model.graph.initializer
                if initializer.name == scale_name
            )
            bias_name = f"{node.name}_omg_zero_bias"
            if bias_name not in initializer_names:
                bias = numpy_helper.from_array(
                    torch.zeros(tuple(scale_initializer.dims), dtype=torch.float32).numpy(),
                    name=bias_name,
                )
                model.graph.initializer.append(bias)
                initializer_names.add(bias_name)
            node.input.append(bias_name)
        if node.op_type.startswith("Reduce"):
            attributes = [
                attribute
                for attribute in node.attribute
                if attribute.name != "noop_with_empty_axes"
            ]
            if len(attributes) != len(node.attribute):
                del node.attribute[:]
                node.attribute.extend(attributes)
        if node.op_type != "Reshape":
            continue
        attributes = [attribute for attribute in node.attribute if attribute.name != "allowzero"]
        if len(attributes) == len(node.attribute):
            continue
        del node.attribute[:]
        node.attribute.extend(attributes)
    expand_clip_scalar_bounds_for_omg(model)
    rewrite_double_casts_for_omg(model)
    rewrite_rank3_transpose_for_omg(model)
    rewrite_clip_for_omg(model)
    rewrite_static_squeeze_ops_for_omg(model)
    duplicate_shared_small_initializers_for_omg(model)
    remove_external_data_file(onnx_path)
    mark_model_sanitized(model)
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{onnx_path.name}.data",
        size_threshold=1024,
    )


def quantize_onnx(cfg: DictConfig, onnx_path: Path, output_dir: Path) -> Path | None:
    mode = str(cfg.export.quantization.mode)
    if mode not in SUPPORTED_QUANTIZATION_MODES:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    if mode == "none":
        return None
    if mode == "int4_palette":
        raise NotImplementedError(
            "int4_palette is intentionally gated until the target CANN toolchain accepts "
            "a specific 4-bit ONNX representation. Use export.quantization.mode=int8_dynamic "
            "for the first CANN conversion pass."
        )

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantization_input = strip_intermediate_value_info(onnx_path, output_dir)
    quantized_path = quantized_output_path(cfg, output_dir)
    quantize_dynamic(
        model_input=str(quantization_input),
        model_output=str(quantized_path),
        per_channel=bool(cfg.export.quantization.per_channel),
        reduce_range=bool(cfg.export.quantization.reduce_range),
        weight_type=QuantType.QInt8,
        use_external_data_format=True,
    )
    return quantized_path


def use_quantized_onnx_for_omg(cfg: DictConfig) -> bool:
    return bool(cfg.export.cann.get("use_quantized_onnx", True))


def cann_target(cfg: DictConfig) -> str:
    target = str(cfg.export.cann.get("target", "omc"))
    if target not in SUPPORTED_CANN_TARGETS:
        raise ValueError(f"Unsupported CANN target: {target}")
    return target


def optional_omg_flag(cfg: DictConfig, key: str) -> list[str]:
    value = none_if_null(cfg.export.cann.get(key))
    if value is None:
        return []
    return [f"--{key}={value}"]


def omg_command(
    cfg: DictConfig, onnx_path: Path, shape: Flux2DenoiserShape, output_dir: Path
) -> list[str]:
    cann_cfg = cfg.export.cann
    output_base = output_dir / str(cann_cfg.output_name)
    model_arg = (
        onnx_path.relative_to(REPO_ROOT) if onnx_path.is_relative_to(REPO_ROOT) else onnx_path
    )
    output_arg = (
        output_base.relative_to(REPO_ROOT) if output_base.is_relative_to(REPO_ROOT) else output_base
    )
    input_shape = ";".join(
        [
            f"hidden_states:{shape.batch_size},{shape.denoiser_tokens},{shape.packed_latent_channels}",
            f"timestep:{shape.batch_size}",
            f"guidance:{shape.batch_size}",
            (
                "encoder_hidden_states:"
                f"{shape.batch_size},{shape.max_sequence_length},{shape.prompt_embed_dim}"
            ),
            f"txt_ids:{shape.batch_size},{shape.max_sequence_length},4",
            f"img_ids:{shape.batch_size},{shape.denoiser_tokens},4",
        ]
    )
    return [
        str(cann_cfg.omg_path),
        "--framework=5",
        f"--model={model_arg}",
        f"--output={output_arg}",
        f"--target={cann_target(cfg)}",
        f"--platform={cann_cfg.platform}",
        f"--input_shape={input_shape}",
        *optional_omg_flag(cfg, "input_format"),
        *optional_omg_flag(cfg, "weight_data_type"),
        *optional_omg_flag(cfg, "input_type"),
        *optional_omg_flag(cfg, "output_type"),
    ]


def om_output_path(cfg: DictConfig, output_dir: Path) -> Path:
    return output_dir / f"{cfg.export.cann.output_name}.{cann_target(cfg)}"


def conversion_input_path(onnx_path: Path, quantized_path: Path | None) -> Path:
    if quantized_path is not None:
        return quantized_path
    return onnx_path


def existing_quantized_model_path(cfg: DictConfig, output_dir: Path) -> Path | None:
    if str(cfg.export.quantization.mode) == "none":
        return None
    quantized_path = quantized_output_path(cfg, output_dir)
    if quantized_path.exists():
        return quantized_path
    return None


def om_conversion_input_path(cfg: DictConfig, onnx_path: Path, quantized_path: Path | None) -> Path:
    if use_quantized_onnx_for_omg(cfg):
        return conversion_input_path(onnx_path, quantized_path)
    return onnx_path


def prepare_om_conversion_model(cfg: DictConfig, onnx_path: Path, output_dir: Path) -> Path:
    if not bool(cfg.export.cann.get("use_shape_stripped_onnx", False)):
        return onnx_path
    stripped_path = strip_intermediate_value_info(onnx_path, output_dir)
    sanitize_onnx_for_omg(stripped_path)
    return stripped_path


def expected_om_conversion_model(
    cfg: DictConfig, onnx_path: Path, quantized_path: Path | None, output_dir: Path
) -> Path:
    conversion_model = om_conversion_input_path(cfg, onnx_path, quantized_path)
    if conversion_model != onnx_path or not bool(
        cfg.export.cann.get("use_shape_stripped_onnx", False)
    ):
        return conversion_model
    return output_dir / f"{onnx_path.stem}.shape_stripped{onnx_path.suffix}"


def require_omg(omg_path: str) -> str:
    configured_path = Path(omg_path)
    resolved: str | None
    if configured_path.exists():
        resolved = str(configured_path)
    else:
        resolved = shutil.which(omg_path)
    if resolved is None:
        raise FileNotFoundError(
            "HarmonyOS CANN Kit OMG executable was not found. Download DDK-tools from "
            "Huawei CANN Kit preparations, install the target platform plugin under "
            "tools/platform/<platform>, or set export.cann.omg_path to tools_omg/omg."
        )
    return resolved


def cann_env_script(cfg: DictConfig) -> Path | None:
    configured = none_if_null(cfg.export.cann.get("env_script"))
    if configured is not None:
        return resolve_repo_path(configured)
    omg_path = Path(str(cfg.export.cann.omg_path))
    ddk_tools_path = omg_path.parents[1] if len(omg_path.parents) > 1 else Path("tools")
    candidate = ddk_tools_path / "tools_ascendc" / "set_ascendc_env.sh"
    if candidate.exists():
        return candidate
    return None


def command_with_cann_env(cfg: DictConfig, command: list[str]) -> list[str]:
    env_script = cann_env_script(cfg)
    if env_script is None:
        return command
    package_path = env_script.parent / "package"
    quoted_command = " ".join(shlex.quote(part) for part in command)
    path_prefix = f"{shlex.quote(str(package_path))}:/usr/bin:/bin"
    ld_prefix = "/usr/lib/x86_64-linux-gnu"
    script = (
        f"source {shlex.quote(str(env_script))} >/dev/null && "
        f"export PATH={path_prefix}:$PATH && "
        f"export LD_LIBRARY_PATH={ld_prefix}:$LD_LIBRARY_PATH && "
        f"{quoted_command}"
    )
    return ["bash", "-lc", script]


def cann_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    return env


def convert_to_om(
    cfg: DictConfig,
    onnx_path: Path,
    quantized_path: Path | None,
    shape: Flux2DenoiserShape,
    output_dir: Path,
) -> Path:
    omg_executable = require_omg(str(cfg.export.cann.omg_path))
    conversion_model = om_conversion_input_path(cfg, onnx_path, quantized_path)
    if conversion_model == onnx_path:
        conversion_model = prepare_om_conversion_model(cfg, onnx_path, output_dir)
    command = omg_command(cfg, conversion_model, shape, output_dir)
    command[0] = omg_executable
    try:
        subprocess.run(
            command_with_cann_env(cfg, command),
            cwd=REPO_ROOT,
            check=True,
            env=cann_subprocess_env(),
        )
    except subprocess.CalledProcessError:
        if (
            quantized_path is None
            or not use_quantized_onnx_for_omg(cfg)
            or not bool(cfg.export.cann.fallback_to_fp16)
        ):
            raise
        fp16_command = omg_command(cfg, onnx_path, shape, output_dir)
        if bool(cfg.export.cann.get("use_shape_stripped_onnx", False)):
            fp16_command = omg_command(
                cfg,
                prepare_om_conversion_model(cfg, onnx_path, output_dir),
                shape,
                output_dir,
            )
        fp16_command[0] = omg_executable
        subprocess.run(
            command_with_cann_env(cfg, fp16_command),
            cwd=REPO_ROOT,
            check=True,
            env=cann_subprocess_env(),
        )
    return om_output_path(cfg, output_dir)


def write_manifest(
    cfg: DictConfig,
    output_dir: Path,
    checkpoint_dir: Path,
    onnx_path: Path,
    quantized_path: Path | None,
    shape: Flux2DenoiserShape,
) -> Path:
    manifest_path = output_dir / str(cfg.export.manifest_filename)
    manifest = {
        "model_key": str(cfg.model_key),
        "checkpoint_dir": str(checkpoint_dir.relative_to(REPO_ROOT))
        if checkpoint_dir.is_relative_to(REPO_ROOT)
        else str(checkpoint_dir),
        "onnx_path": str(onnx_path.relative_to(REPO_ROOT))
        if onnx_path.is_relative_to(REPO_ROOT)
        else str(onnx_path),
        "quantized_onnx_path": None
        if quantized_path is None
        else (
            str(quantized_path.relative_to(REPO_ROOT))
            if quantized_path.is_relative_to(REPO_ROOT)
            else str(quantized_path)
        ),
        "shape": asdict(shape),
        "input_names": ONNX_INPUT_NAMES,
        "output_names": ONNX_OUTPUT_NAMES,
        "quantization": OmegaConf.to_container(cfg.export.quantization, resolve=True),
        "cann_omg_command": omg_command(
            cfg,
            expected_om_conversion_model(cfg, onnx_path, quantized_path, output_dir),
            shape,
            output_dir,
        ),
        "om_path": str(om_output_path(cfg, output_dir).relative_to(REPO_ROOT))
        if om_output_path(cfg, output_dir).is_relative_to(REPO_ROOT)
        else str(om_output_path(cfg, output_dir)),
        "lora_fused": True,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def run(cfg: DictConfig) -> Path:
    configure_environment(cfg)
    output_dir = resolve_repo_path(str(cfg.export.output_root))
    checkpoint_dir = checkpoint_dir_from_config(cfg)
    dtype = dtype_from_precision(str(cfg.export.precision))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = load_fused_transformer(cfg, checkpoint_dir).to(device)
    shape = shape_with_transformer_config(shape_from_config(cfg), transformer)
    wrapper = Flux2DenoiserExportWrapper(transformer)
    inputs = dummy_inputs(shape, dtype, device)
    onnx_path = output_dir / str(cfg.export.onnx_filename)
    export_onnx(wrapper, inputs, onnx_path, int(cfg.export.opset))
    sanitize_onnx_for_omg(onnx_path)
    quantized_path = quantize_onnx(cfg, onnx_path, output_dir)
    if quantized_path is not None:
        sanitize_onnx_for_omg(quantized_path)
    manifest_path = write_manifest(
        cfg, output_dir, checkpoint_dir, onnx_path, quantized_path, shape
    )
    if bool(cfg.export.cann.convert_om):
        convert_to_om(cfg, onnx_path, quantized_path, shape, output_dir)
    return manifest_path


def convert_existing_om(cfg: DictConfig) -> Path:
    output_dir = resolve_repo_path(str(cfg.export.output_root))
    onnx_path = output_dir / str(cfg.export.onnx_filename)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing exported ONNX model: {onnx_path}")
    quantized_model = existing_quantized_model_path(cfg, output_dir)
    sanitize_onnx_for_omg(onnx_path)
    if quantized_model is not None:
        sanitize_onnx_for_omg(quantized_model)
    shape = shape_from_config(cfg)
    checkpoint_dir = checkpoint_dir_from_config(cfg)
    write_manifest(cfg, output_dir, checkpoint_dir, onnx_path, quantized_model, shape)
    return convert_to_om(cfg, onnx_path, quantized_model, shape, output_dir)


def main() -> None:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="export_flux2_cann", overrides=sys.argv[1:])
    manifest_path = run(cfg)
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))


def convert_existing_om_main() -> None:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="export_flux2_cann", overrides=sys.argv[1:])
    try:
        om_path = convert_existing_om(cfg)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({"om": str(om_path)}, indent=2))


__all__ = [
    "Flux2DenoiserExportWrapper",
    "Flux2DenoiserInputs",
    "Flux2DenoiserShape",
    "ONNX_INPUT_NAMES",
    "ONNX_OUTPUT_NAMES",
    "ExportableRMSNorm",
    "checkpoint_dir_from_config",
    "convert_existing_om",
    "convert_existing_om_main",
    "conversion_input_path",
    "existing_quantized_model_path",
    "om_conversion_input_path",
    "prepare_om_conversion_model",
    "expected_om_conversion_model",
    "convert_to_om",
    "dummy_inputs",
    "export_onnx",
    "load_fused_transformer",
    "main",
    "quantize_onnx",
    "omg_command",
    "require_omg",
    "replace_rms_norm_modules",
    "run",
    "sanitize_onnx_for_omg",
    "shape_from_config",
    "shape_with_transformer_config",
    "strip_intermediate_value_info",
    "validate_shape",
    "write_manifest",
]
