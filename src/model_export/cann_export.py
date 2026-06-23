from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import onnx
import torch
from onnx import TensorProto, numpy_helper
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from lora.local_edit_common import REPO_ROOT, none_if_null, resolve_repo_path


SUPPORTED_QUANTIZATION_MODES = {"none", "int8_dynamic", "int8_dopt"}
SUPPORTED_CANN_TARGETS = {"om", "omc", "tiny", "ispnn", "security"}
SANITIZER_EXTERNAL_DATA_MARKER = "cann_omg_sanitized"
SANITIZER_VERSION = "7"
MAX_DUPLICATED_INITIALIZER_BYTES = 1024

InputShapes = Mapping[str, Sequence[int]]


class InputShapeProvider(Protocol):
    @property
    def input_shapes(self) -> InputShapes: ...


@dataclass(frozen=True)
class CannModelSpec:
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_shapes: InputShapes
    metadata: Mapping[str, Any] = field(default_factory=dict)
    model: nn.Module | None = None
    inputs: tuple[Tensor, ...] = ()


def export_onnx(spec: CannModelSpec, output_path: Path, opset: int) -> None:
    if spec.model is None:
        raise ValueError("CannModelSpec.model is required for ONNX export")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec.model.eval()
    with torch.no_grad():
        torch.onnx.export(
            spec.model,
            spec.inputs,
            output_path,
            input_names=list(spec.input_names),
            output_names=list(spec.output_names),
            opset_version=opset,
            do_constant_folding=True,
        )


def quantized_output_path(cfg: DictConfig, output_dir: Path) -> Path:
    return output_dir / str(cfg.export.quantized_onnx_filename)


def input_shape_argument(model: InputShapeProvider) -> str:
    shapes = model.input_shapes
    return ";".join(
        f"{name}:{','.join(str(dimension) for dimension in shape)}"
        for name, shape in shapes.items()
    )


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


def sanitize_onnx_for_omg(onnx_path: Path, *, force: bool = False) -> None:
    model = onnx.load(onnx_path, load_external_data=True)
    if model_is_marked_sanitized(model) and not force:
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


def dopt_calibration_config_path(cfg: DictConfig, output_dir: Path) -> Path:
    configured = none_if_null(cfg.export.quantization.get("calibration_config"))
    if configured is not None:
        return resolve_repo_path(configured)
    config_path = output_dir / "dopt_int8_calibration.prototxt"
    config_path.write_text(
        "\n".join(
            [
                "strategy: 'Quant_INT8-8'",
                "device: USE_CPU",
                "preprocess_parameter:",
                "{",
                "    input_type: BIN",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def dopt_compress_config_path(cfg: DictConfig, output_dir: Path) -> Path:
    configured = none_if_null(cfg.export.quantization.get("compress_config"))
    if configured is not None:
        return resolve_repo_path(configured)
    return output_dir / "dopt_int8_params"


def dopt_python_executable(cfg: DictConfig) -> str:
    configured = none_if_null(cfg.export.quantization.get("dopt_python"))
    if configured is not None:
        return str(configured)
    return os.environ.get("CANN_DOPT_PYTHON", "python3")


def dopt_executable_path(cfg: DictConfig) -> Path:
    configured = none_if_null(cfg.export.quantization.get("dopt_path"))
    if configured is not None:
        return resolve_repo_path(configured)
    return resolve_repo_path(".cannkit_tools/ddk/tools/tools_dopt/dopt_onnx_py3/dopt_so.py")


def dopt_command(
    cfg: DictConfig,
    onnx_path: Path,
    quantized_path: Path,
    output_dir: Path,
    model: InputShapeProvider,
) -> list[str]:
    return [
        dopt_python_executable(cfg),
        str(dopt_executable_path(cfg)),
        "--framework",
        "5",
        "-m",
        "0",
        "--model",
        str(onnx_path),
        "--cal_conf",
        str(dopt_calibration_config_path(cfg, output_dir)),
        "--output",
        str(quantized_path),
        "--input_shape",
        input_shape_argument(model),
        "--compress_conf",
        str(dopt_compress_config_path(cfg, output_dir)),
        "--device_idx",
        str(int(cfg.export.quantization.get("device_idx", 0))),
    ]


def quantize_onnx(
    cfg: DictConfig,
    onnx_path: Path,
    output_dir: Path,
    model: InputShapeProvider,
) -> Path | None:
    mode = str(cfg.export.quantization.mode)
    if mode not in SUPPORTED_QUANTIZATION_MODES:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    if mode == "none":
        return None
    if mode == "int8_dopt":
        quantized_path = quantized_output_path(cfg, output_dir)
        subprocess.run(
            dopt_command(cfg, onnx_path, quantized_path, output_dir, model),
            cwd=REPO_ROOT,
            check=True,
            env=cann_subprocess_env(),
        )
        return quantized_path

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantization_input = strip_intermediate_value_info(onnx_path, output_dir)
    quantized_path = quantized_output_path(cfg, output_dir)
    quantize_dynamic(
        model_input=str(quantization_input),
        model_output=str(quantized_path),
        per_channel=bool(cfg.export.quantization.get("per_channel", True)),
        reduce_range=bool(cfg.export.quantization.get("reduce_range", False)),
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
    cfg: DictConfig, onnx_path: Path, model: InputShapeProvider, output_dir: Path
) -> list[str]:
    cann_cfg = cfg.export.cann
    output_base = output_dir / str(cann_cfg.output_name)
    model_arg = (
        onnx_path.relative_to(REPO_ROOT) if onnx_path.is_relative_to(REPO_ROOT) else onnx_path
    )
    output_arg = (
        output_base.relative_to(REPO_ROOT) if output_base.is_relative_to(REPO_ROOT) else output_base
    )
    return [
        str(cann_cfg.omg_path),
        "--framework=5",
        f"--model={model_arg}",
        f"--output={output_arg}",
        f"--target={cann_target(cfg)}",
        f"--platform={cann_cfg.platform}",
        f"--input_shape={input_shape_argument(model)}",
        *optional_omg_flag(cfg, "input_format"),
        *optional_omg_flag(cfg, "weight_data_type"),
        *optional_omg_flag(cfg, "input_type"),
        *optional_omg_flag(cfg, "output_type"),
    ]


def om_output_path(cfg: DictConfig, output_dir: Path) -> Path:
    return output_dir / f"{cfg.export.cann.output_name}.{cann_target(cfg)}"


def quantization_mode(cfg: DictConfig) -> str:
    quantization_cfg = cfg.export.get("quantization")
    if quantization_cfg is None:
        return "none"
    return str(quantization_cfg.get("mode", "none"))


def contains_onnxruntime_dynamic_int8_ops(onnx_path: Path) -> bool:
    if not onnx_path.exists():
        return False
    model = onnx.load(onnx_path, load_external_data=False)
    unsupported_ops = {"DynamicQuantizeLinear", "MatMulInteger"}
    return any(node.op_type in unsupported_ops for node in model.graph.node)


def validate_omg_quantized_input(cfg: DictConfig, quantized_path: Path | None) -> None:
    if quantized_path is None:
        return
    if quantization_mode(cfg) == "int8_dynamic":
        raise ValueError(
            "export.quantization.mode=int8_dynamic creates ONNX Runtime "
            "DynamicQuantizeLinear/MatMulInteger graphs, which this CANN Kit OMG pre-check "
            "does not support. Use export.quantization.mode=int8_dopt for CANN INT8 OM "
            "conversion, or set export.cann.use_quantized_onnx=false to convert the FP16 ONNX."
        )
    if contains_onnxruntime_dynamic_int8_ops(quantized_path):
        raise ValueError(
            f"Existing quantized ONNX contains ONNX Runtime dynamic INT8 operators unsupported "
            f"by this CANN Kit OMG: {quantized_path}. Remove the stale artifact and regenerate "
            "with export.quantization.mode=int8_dopt, or set export.cann.use_quantized_onnx=false."
        )


def om_conversion_input_path(cfg: DictConfig, onnx_path: Path, quantized_path: Path | None) -> Path:
    if use_quantized_onnx_for_omg(cfg):
        validate_omg_quantized_input(cfg, quantized_path)
        return onnx_path if quantized_path is None else quantized_path
    return onnx_path


def prepare_om_conversion_model(cfg: DictConfig, onnx_path: Path, output_dir: Path) -> Path:
    if not bool(cfg.export.cann.get("use_shape_stripped_onnx", False)):
        return onnx_path
    stripped_path = strip_intermediate_value_info(onnx_path, output_dir)
    sanitize_onnx_for_omg(stripped_path)
    return stripped_path


def require_omg(omg_path: str) -> str:
    configured_path = resolve_repo_path(omg_path)
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
    omg_path = resolve_repo_path(str(cfg.export.cann.omg_path))
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
    model: InputShapeProvider,
    output_dir: Path,
) -> tuple[Path, Path]:
    omg_executable = require_omg(str(cfg.export.cann.omg_path))
    conversion_model = om_conversion_input_path(cfg, onnx_path, quantized_path)
    if conversion_model == onnx_path:
        conversion_model = prepare_om_conversion_model(cfg, onnx_path, output_dir)
    command = omg_command(cfg, conversion_model, model, output_dir)
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
        conversion_model = prepare_om_conversion_model(cfg, onnx_path, output_dir)
        fp16_command = omg_command(cfg, conversion_model, model, output_dir)
        fp16_command[0] = omg_executable
        subprocess.run(
            command_with_cann_env(cfg, fp16_command),
            cwd=REPO_ROOT,
            check=True,
            env=cann_subprocess_env(),
        )
    return om_output_path(cfg, output_dir), conversion_model


def relative_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def write_manifest(
    cfg: DictConfig,
    output_dir: Path,
    onnx_path: Path,
    quantized_path: Path | None,
    spec: CannModelSpec,
    om_path: Path | None,
    conversion_model: Path | None,
) -> Path:
    manifest_path = output_dir / str(cfg.export.manifest_filename)
    manifest = {
        **spec.metadata,
        "onnx_path": relative_path(onnx_path),
        "quantized_onnx_path": (None if quantized_path is None else relative_path(quantized_path)),
        "input_names": list(spec.input_names),
        "output_names": list(spec.output_names),
        "quantization": OmegaConf.to_container(cfg.export.get("quantization", {}), resolve=True),
        "cann_omg_command": (
            None
            if conversion_model is None
            else omg_command(cfg, conversion_model, spec, output_dir)
        ),
        "om_path": None if om_path is None else relative_path(om_path),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def run_export_pipeline(cfg: DictConfig, output_dir: Path, spec: CannModelSpec) -> Path:
    onnx_path = output_dir / str(cfg.export.onnx_filename)
    export_onnx(spec, onnx_path, int(cfg.export.opset))
    sanitize_onnx_for_omg(onnx_path)
    quantized_path = quantize_onnx(cfg, onnx_path, output_dir, spec)
    if quantized_path is not None:
        sanitize_onnx_for_omg(quantized_path, force=True)
    om_path: Path | None = None
    conversion_model: Path | None = None
    if bool(cfg.export.cann.convert_om):
        om_path, conversion_model = convert_to_om(cfg, onnx_path, quantized_path, spec, output_dir)
    return write_manifest(
        cfg,
        output_dir,
        onnx_path,
        quantized_path,
        spec,
        om_path,
        conversion_model,
    )
