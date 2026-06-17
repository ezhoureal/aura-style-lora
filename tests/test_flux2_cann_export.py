from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import onnx
from onnx import TensorProto, helper
from torch import nn
from omegaconf import OmegaConf

from lora.flux2_cann_export import (
    ExportableRMSNorm,
    Flux2DenoiserShape,
    cann_target,
    checkpoint_dir_from_config,
    convert_existing_om,
    conversion_input_path,
    convert_to_om,
    dummy_inputs,
    existing_quantized_model_path,
    expected_om_conversion_model,
    om_conversion_input_path,
    omg_command,
    quantize_onnx,
    require_omg,
    sanitize_onnx_for_omg,
    shape_from_config,
    static_shapes,
    strip_intermediate_value_info,
    write_manifest,
)
from lora.local_edit_common import REPO_ROOT


class Flux2CannExportShapeTests(unittest.TestCase):
    def test_exportable_rms_norm_matches_torch_rms_norm(self) -> None:
        torch_norm = nn.RMSNorm(4, eps=1e-6)
        export_norm = ExportableRMSNorm.from_torch_rms_norm(torch_norm)
        hidden_states = torch.randn(2, 3, 4)

        expected = torch_norm(hidden_states)
        actual = export_norm(hidden_states)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_shape_contract_matches_packed_flux2_edit_inputs(self) -> None:
        cfg = OmegaConf.create(
            {
                "export": {
                    "batch_size": 1,
                    "height": 512,
                    "width": 512,
                    "max_sequence_length": 512,
                    "prompt_embed_dim": 7680,
                    "packed_latent_channels": 128,
                }
            }
        )

        shape = shape_from_config(cfg)
        inputs = dummy_inputs(shape, torch.float16, torch.device("cpu"))

        self.assertEqual(shape.packed_latent_tokens, 1024)
        self.assertEqual(shape.denoiser_tokens, 2048)
        self.assertEqual(inputs.hidden_states.shape, (1, 2048, 128))
        self.assertEqual(inputs.timestep.shape, (1,))
        self.assertEqual(inputs.guidance.shape, (1,))
        self.assertEqual(inputs.encoder_hidden_states.shape, (1, 512, 7680))
        self.assertEqual(inputs.txt_ids.shape, (1, 512, 4))
        self.assertEqual(inputs.img_ids.shape, (1, 2048, 4))

    def test_shape_rejects_non_flux_latent_multiple(self) -> None:
        cfg = OmegaConf.create(
            {
                "export": {
                    "batch_size": 1,
                    "height": 510,
                    "width": 512,
                    "max_sequence_length": 512,
                    "prompt_embed_dim": 4096,
                    "packed_latent_channels": 64,
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            shape_from_config(cfg)


class Flux2CannExportManifestTests(unittest.TestCase):
    def test_checkpoint_dir_must_be_configured(self) -> None:
        cfg = OmegaConf.create({"export": {"checkpoint_dir": None}})

        with self.assertRaisesRegex(ValueError, "export.checkpoint_dir"):
            checkpoint_dir_from_config(cfg)

    def test_manifest_records_relative_paths_and_omg_command(self) -> None:
        cfg = OmegaConf.create(
            {
                "model_key": "flux2_klein_base",
                "export": {
                    "manifest_filename": "manifest.json",
                    "quantization": {
                        "mode": "int8_dynamic",
                        "per_channel": True,
                        "reduce_range": False,
                    },
                    "cann": {
                        "platform": "kirin9030",
                        "output_name": "transformer_denoiser",
                        "omg_path": "omg",
                        "target": "omc",
                        "input_format": None,
                        "weight_data_type": "FP16",
                        "fallback_to_fp16": True,
                    },
                },
            }
        )
        shape = Flux2DenoiserShape(
            batch_size=1,
            height=512,
            width=512,
            max_sequence_length=512,
            prompt_embed_dim=7680,
            packed_latent_channels=128,
        )
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            output_dir = Path(tmp) / "export"
            output_dir.mkdir()
            checkpoint_dir = Path(tmp) / "checkpoint-000001"
            checkpoint_dir.mkdir()
            onnx_path = output_dir / "transformer.onnx"
            quantized_path = output_dir / "transformer.int8.onnx"

            manifest_path = write_manifest(
                cfg,
                output_dir,
                checkpoint_dir,
                onnx_path,
                quantized_path,
                shape,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertFalse(manifest["checkpoint_dir"].startswith("/home/"))
        self.assertFalse(manifest["onnx_path"].startswith("/home/"))
        self.assertEqual(manifest["quantization"]["mode"], "int8_dynamic")
        self.assertIn("--framework=5", manifest["cann_omg_command"])
        self.assertIn("--target=omc", manifest["cann_omg_command"])
        self.assertIn("--platform=kirin9030", manifest["cann_omg_command"])
        self.assertNotIn("--input_format=None", manifest["cann_omg_command"])
        self.assertIn("--weight_data_type=FP16", manifest["cann_omg_command"])
        self.assertTrue(
            any("hidden_states:1,2048,128" in arg for arg in manifest["cann_omg_command"])
        )
        self.assertEqual(
            manifest["om_path"],
            f"{output_dir.relative_to(REPO_ROOT)}/transformer_denoiser.omc",
        )

    def test_omg_command_uses_quantized_model_when_supplied_by_manifest_helper(self) -> None:
        cfg = OmegaConf.create(
            {
                "export": {
                    "cann": {
                        "platform": "kirin9030",
                        "output_name": "transformer_denoiser",
                        "omg_path": "omg",
                        "target": "omc",
                    }
                }
            }
        )
        shape = Flux2DenoiserShape(
            batch_size=1,
            height=512,
            width=512,
            max_sequence_length=512,
            prompt_embed_dim=7680,
            packed_latent_channels=128,
        )

        command = omg_command(cfg, Path("model.int8.onnx"), shape, Path("out"))

        self.assertIn("--model=model.int8.onnx", command)
        self.assertIn("--output=out/transformer_denoiser", command)
        self.assertIn("--target=omc", command)
        self.assertIn("--platform=kirin9030", command)
        self.assertNotIn("--input_format=NCHW", command)

    def test_manifest_records_fp16_omg_input_when_quantized_conversion_is_disabled(self) -> None:
        cfg = OmegaConf.create(
            {
                "model_key": "flux2_klein_base",
                "export": {
                    "manifest_filename": "manifest.json",
                    "quantization": {
                        "mode": "int8_dynamic",
                        "per_channel": True,
                        "reduce_range": False,
                    },
                    "cann": {
                        "platform": "kirin9030",
                        "output_name": "transformer_denoiser",
                        "omg_path": "omg",
                        "target": "omc",
                        "use_quantized_onnx": False,
                    },
                },
            }
        )
        shape = Flux2DenoiserShape(
            batch_size=1,
            height=512,
            width=512,
            max_sequence_length=512,
            prompt_embed_dim=7680,
            packed_latent_channels=128,
        )
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            output_dir = Path(tmp) / "export"
            output_dir.mkdir()
            checkpoint_dir = Path(tmp) / "checkpoint-000001"
            checkpoint_dir.mkdir()
            manifest_path = write_manifest(
                cfg,
                output_dir,
                checkpoint_dir,
                output_dir / "transformer.onnx",
                output_dir / "transformer.int8.onnx",
                shape,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertTrue(manifest["cann_omg_command"][2].endswith("transformer.onnx"))

    def test_cann_target_rejects_unknown_target(self) -> None:
        cfg = OmegaConf.create({"export": {"cann": {"target": "bad_target"}}})

        with self.assertRaisesRegex(ValueError, "Unsupported CANN target"):
            cann_target(cfg)

    def test_conversion_prefers_quantized_model_when_available(self) -> None:
        self.assertEqual(
            conversion_input_path(Path("model.onnx"), Path("model.int8.onnx")),
            Path("model.int8.onnx"),
        )
        self.assertEqual(conversion_input_path(Path("model.onnx"), None), Path("model.onnx"))

    def test_om_conversion_can_skip_quantized_model_for_mobile_omg(self) -> None:
        cfg = OmegaConf.create({"export": {"cann": {"use_quantized_onnx": False}}})

        self.assertEqual(
            om_conversion_input_path(cfg, Path("model.onnx"), Path("model.int8.onnx")),
            Path("model.onnx"),
        )

    def test_expected_om_conversion_model_uses_shape_stripped_artifact(self) -> None:
        cfg = OmegaConf.create(
            {
                "export": {
                    "cann": {
                        "use_quantized_onnx": False,
                        "use_shape_stripped_onnx": True,
                    }
                }
            }
        )

        self.assertEqual(
            expected_om_conversion_model(cfg, Path("model.onnx"), None, Path("out")),
            Path("out/model.shape_stripped.onnx"),
        )

    def test_require_omg_reports_missing_toolchain(self) -> None:
        with mock.patch("lora.flux2_cann_export.shutil.which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "CANN Kit OMG executable"):
                require_omg("omg")

    def test_convert_to_om_falls_back_to_fp16_when_quantized_omg_fails(self) -> None:
        cfg = OmegaConf.create(
            {
                "export": {
                    "cann": {
                        "platform": "kirin9030",
                        "output_name": "transformer_denoiser",
                        "omg_path": "omg",
                        "fallback_to_fp16": True,
                        "use_quantized_onnx": True,
                    }
                }
            }
        )
        shape = Flux2DenoiserShape(
            batch_size=1,
            height=512,
            width=512,
            max_sequence_length=512,
            prompt_embed_dim=7680,
            packed_latent_channels=128,
        )

        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            output_dir = Path(tmp)
            with (
                mock.patch("lora.flux2_cann_export.require_omg", return_value="/opt/omg"),
                mock.patch("lora.flux2_cann_export.cann_env_script", return_value=None),
                mock.patch("lora.flux2_cann_export.subprocess.run") as run_mock,
            ):
                run_mock.side_effect = [
                    subprocess.CalledProcessError(1, ["omg"]),
                    subprocess.CompletedProcess(["omg"], 0),
                ]

                om_path = convert_to_om(
                    cfg,
                    Path("model.onnx"),
                    Path("model.int8.onnx"),
                    shape,
                    output_dir,
                )

        self.assertEqual(om_path.name, "transformer_denoiser.omc")
        self.assertEqual(run_mock.call_count, 2)
        self.assertIn("--model=model.int8.onnx", run_mock.call_args_list[0].args[0])
        self.assertIn("--model=model.onnx", run_mock.call_args_list[1].args[0])
        self.assertIn("--target=omc", run_mock.call_args_list[0].args[0])

    def test_convert_to_om_uses_fp16_when_quantized_omg_input_is_disabled(self) -> None:
        cfg = OmegaConf.create(
            {
                "export": {
                    "cann": {
                        "platform": "kirin9030",
                        "output_name": "transformer_denoiser",
                        "omg_path": "omg",
                        "fallback_to_fp16": True,
                        "use_quantized_onnx": False,
                    }
                }
            }
        )
        shape = Flux2DenoiserShape(
            batch_size=1,
            height=512,
            width=512,
            max_sequence_length=512,
            prompt_embed_dim=7680,
            packed_latent_channels=128,
        )

        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            with (
                mock.patch("lora.flux2_cann_export.require_omg", return_value="/opt/omg"),
                mock.patch("lora.flux2_cann_export.cann_env_script", return_value=None),
                mock.patch("lora.flux2_cann_export.subprocess.run") as run_mock,
            ):
                convert_to_om(
                    cfg,
                    Path("model.onnx"),
                    Path("model.int8.onnx"),
                    shape,
                    Path(tmp),
                )

        self.assertEqual(run_mock.call_count, 1)
        self.assertIn("--model=model.onnx", run_mock.call_args.args[0])

    def test_convert_existing_om_uses_existing_quantized_artifact(self) -> None:
        shape_cfg = {
            "batch_size": 1,
            "height": 512,
            "width": 512,
            "max_sequence_length": 512,
            "prompt_embed_dim": 7680,
            "packed_latent_channels": 128,
        }
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp)
            output_dir = root / "export"
            checkpoint_dir = root / "checkpoint-000001"
            output_dir.mkdir()
            checkpoint_dir.mkdir()
            (output_dir / "transformer.onnx").touch()
            (output_dir / "transformer.int8.onnx").touch()
            cfg = OmegaConf.create(
                {
                    "model_key": "flux2_klein_base",
                    "export": {
                        "checkpoint_dir": str(checkpoint_dir.relative_to(REPO_ROOT)),
                        "output_root": str(output_dir.relative_to(REPO_ROOT)),
                        "onnx_filename": "transformer.onnx",
                        "quantized_onnx_filename": "transformer.int8.onnx",
                        "manifest_filename": "manifest.json",
                        **shape_cfg,
                        "quantization": {
                            "mode": "int8_dynamic",
                            "per_channel": True,
                            "reduce_range": False,
                        },
                        "cann": {
                            "platform": "kirin9030",
                            "output_name": "transformer_denoiser",
                            "omg_path": "omg",
                            "target": "omc",
                            "fallback_to_fp16": True,
                        },
                    },
                }
            )

            with mock.patch(
                "lora.flux2_cann_export.convert_to_om",
                return_value=output_dir / "transformer_denoiser.omc",
            ) as convert_mock:
                om_path = convert_existing_om(cfg)

        self.assertEqual(om_path.name, "transformer_denoiser.omc")
        self.assertEqual(convert_mock.call_args.args[2].name, "transformer.int8.onnx")

    def test_existing_quantized_model_is_ignored_when_quantization_is_disabled(self) -> None:
        cfg = OmegaConf.create(
            {
                "export": {
                    "quantized_onnx_filename": "transformer.int8.onnx",
                    "quantization": {"mode": "none"},
                }
            }
        )
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            output_dir = Path(tmp)
            (output_dir / "transformer.int8.onnx").touch()

            self.assertIsNone(existing_quantized_model_path(cfg, output_dir))


class Flux2CannQuantizationTests(unittest.TestCase):
    def test_sanitize_onnx_for_omg_removes_reshape_allowzero_attributes(self) -> None:
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
        output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        shape_initializer = helper.make_tensor("shape", TensorProto.INT64, [2], [1, 2])
        node = helper.make_node("Reshape", ["x", "shape"], ["y"], allowzero=1)
        graph = helper.make_graph(
            [node],
            "tiny",
            [input_info],
            [output_info],
            initializer=[shape_initializer],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            model_path = Path(tmp) / "tiny.onnx"
            onnx.save(model, model_path)

            sanitize_onnx_for_omg(model_path)
            sanitized = onnx.load(model_path)

        self.assertEqual(len(sanitized.graph.node[0].attribute), 0)

    def test_sanitize_onnx_for_omg_downcasts_double_initializers(self) -> None:
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
        output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
        weight = helper.make_tensor("scale", TensorProto.DOUBLE, [2], [1.0, 2.0])
        node = helper.make_node("Mul", ["x", "scale"], ["y"])
        graph = helper.make_graph(
            [node],
            "tiny",
            [input_info],
            [output_info],
            initializer=[weight],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            model_path = Path(tmp) / "tiny.onnx"
            onnx.save(model, model_path)

            sanitize_onnx_for_omg(model_path)
            sanitized = onnx.load(model_path)

        self.assertEqual(sanitized.graph.initializer[0].data_type, TensorProto.FLOAT)

    def test_sanitize_onnx_for_omg_rewrites_double_casts(self) -> None:
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
        output_info = helper.make_tensor_value_info("y", TensorProto.DOUBLE, [2])
        node = helper.make_node("Cast", ["x"], ["y"], to=TensorProto.DOUBLE)
        graph = helper.make_graph([node], "tiny", [input_info], [output_info])
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            model_path = Path(tmp) / "tiny.onnx"
            onnx.save(model, model_path)

            sanitize_onnx_for_omg(model_path)
            sanitized = onnx.load(model_path)

        cast_to = next(
            attribute.i for attribute in sanitized.graph.node[0].attribute if attribute.name == "to"
        )
        self.assertEqual(cast_to, TensorProto.FLOAT)
        self.assertEqual(sanitized.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT)

    def test_sanitize_onnx_for_omg_adds_layer_norm_bias(self) -> None:
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        scale = helper.make_tensor("scale", TensorProto.FLOAT, [2], [1.0, 1.0])
        node = helper.make_node(
            "LayerNormalization", ["x", "scale"], ["y"], name="layer_norm", axis=-1
        )
        graph = helper.make_graph(
            [node],
            "tiny",
            [input_info],
            [output_info],
            initializer=[scale],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            model_path = Path(tmp) / "tiny.onnx"
            onnx.save(model, model_path)

            sanitize_onnx_for_omg(model_path)
            sanitized = onnx.load(model_path)

        self.assertEqual(len(sanitized.graph.node[0].input), 3)
        self.assertEqual(sanitized.graph.initializer[1].name, "layer_norm_omg_zero_bias")

    def test_sanitize_onnx_for_omg_removes_reduce_noop_with_empty_axes(self) -> None:
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
        axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
        node = helper.make_node(
            "ReduceMean",
            ["x", "axes"],
            ["y"],
            keepdims=1,
            noop_with_empty_axes=0,
        )
        graph = helper.make_graph(
            [node],
            "tiny",
            [input_info],
            [output_info],
            initializer=[axes],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            model_path = Path(tmp) / "tiny.onnx"
            onnx.save(model, model_path)

            sanitize_onnx_for_omg(model_path)
            sanitized = onnx.load(model_path)

        self.assertEqual([attr.name for attr in sanitized.graph.node[0].attribute], ["keepdims"])

    def test_sanitize_onnx_for_omg_expands_clip_scalar_bounds(self) -> None:
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        minimum = helper.make_tensor("minimum", TensorProto.FLOAT, [], [-1.0])
        maximum = helper.make_tensor("maximum", TensorProto.FLOAT, [], [1.0])
        node = helper.make_node("Clip", ["x", "minimum", "maximum"], ["y"])
        graph = helper.make_graph(
            [node],
            "tiny",
            [input_info],
            [output_info],
            initializer=[minimum, maximum],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            model_path = Path(tmp) / "tiny.onnx"
            onnx.save(model, model_path)

            sanitize_onnx_for_omg(model_path)
            sanitized = onnx.load(model_path)

        self.assertEqual(list(sanitized.graph.initializer[0].dims), [1])
        self.assertEqual(list(sanitized.graph.initializer[1].dims), [1])
        self.assertEqual([node.op_type for node in sanitized.graph.node], ["Max", "Min"])

    def test_sanitize_onnx_for_omg_rewrites_rank3_transpose_for_mobile_fusion(self) -> None:
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
        output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2])
        node = helper.make_node("Transpose", ["x"], ["y"], name="rank3_transpose", perm=[0, 2, 1])
        graph = helper.make_graph([node], "tiny", [input_info], [output_info])
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            model_path = Path(tmp) / "tiny.onnx"
            onnx.save(model, model_path)

            sanitize_onnx_for_omg(model_path)
            sanitized = onnx.load(model_path)

        self.assertEqual(
            [node.op_type for node in sanitized.graph.node],
            ["Unsqueeze", "Transpose", "Reshape"],
        )
        transpose = sanitized.graph.node[1]
        self.assertEqual([attribute.name for attribute in transpose.attribute], ["perm"])
        self.assertEqual(list(transpose.attribute[0].ints), [0, 1, 3, 2])

    def test_static_shapes_reads_only_fully_static_shapes(self) -> None:
        static_info = helper.make_tensor_value_info("static", TensorProto.FLOAT, [1, 2])
        dynamic_info = helper.make_tensor_value_info("dynamic", TensorProto.FLOAT, [None, 2])
        graph = helper.make_graph([], "tiny", [static_info, dynamic_info], [])
        model = helper.make_model(graph)

        self.assertEqual(static_shapes(model), {"static": [1, 2]})

    def test_sanitize_onnx_for_omg_rewrites_static_squeeze_ops_to_reshape(self) -> None:
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        unsqueezed_info = helper.make_tensor_value_info("expanded", TensorProto.FLOAT, [1, 2, 1])
        output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        axes = helper.make_tensor("axes", TensorProto.INT64, [1], [-1])
        unsqueeze = helper.make_node("Unsqueeze", ["x", "axes"], ["expanded"], name="expand")
        squeeze = helper.make_node("Squeeze", ["expanded", "axes"], ["y"], name="squeeze")
        graph = helper.make_graph(
            [unsqueeze, squeeze],
            "tiny",
            [input_info],
            [output_info],
            initializer=[axes],
            value_info=[unsqueezed_info],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            model_path = Path(tmp) / "tiny.onnx"
            onnx.save(model, model_path)

            sanitize_onnx_for_omg(model_path)
            sanitized = onnx.load(model_path)

        self.assertEqual([node.op_type for node in sanitized.graph.node], ["Reshape", "Reshape"])
        initializer_names = {initializer.name for initializer in sanitized.graph.initializer}
        self.assertIn("expanded_omg_static_shape", initializer_names)
        self.assertIn("y_omg_static_shape", initializer_names)

    def test_sanitize_onnx_for_omg_duplicates_shared_small_initializers(self) -> None:
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        output_info = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
        shared = helper.make_tensor("shared", TensorProto.FLOAT, [1], [1.0])
        first = helper.make_node("Add", ["x", "shared"], ["y"], name="first")
        second = helper.make_node("Add", ["y", "shared"], ["z"], name="second")
        graph = helper.make_graph(
            [first, second],
            "tiny",
            [input_info],
            [output_info],
            initializer=[shared],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            model_path = Path(tmp) / "tiny.onnx"
            onnx.save(model, model_path)

            sanitize_onnx_for_omg(model_path)
            sanitized = onnx.load(model_path)

        self.assertEqual(sanitized.graph.node[0].input[1], "shared")
        self.assertEqual(sanitized.graph.node[1].input[1], "shared_omg_const_1")
        self.assertIn("shared_omg_const_1", {init.name for init in sanitized.graph.initializer})

    def test_strip_intermediate_value_info_keeps_graph_io_shapes(self) -> None:
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        intermediate_info = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "tiny",
            [input_info],
            [output_info],
            value_info=[intermediate_info],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            model_path = Path(tmp) / "tiny.onnx"
            onnx.save(model, model_path)

            stripped_path = strip_intermediate_value_info(model_path, Path(tmp))
            stripped = onnx.load(stripped_path)

        self.assertEqual(len(stripped.graph.value_info), 0)
        self.assertEqual(stripped.graph.input[0].name, "x")
        self.assertEqual(stripped.graph.output[0].name, "y")

    def test_int4_palette_mode_is_gated_until_cann_representation_is_confirmed(self) -> None:
        cfg = OmegaConf.create(
            {
                "export": {
                    "quantized_onnx_filename": "model.int4.onnx",
                    "quantization": {
                        "mode": "int4_palette",
                        "per_channel": True,
                        "reduce_range": False,
                    },
                }
            }
        )

        with self.assertRaisesRegex(NotImplementedError, "target CANN toolchain"):
            quantize_onnx(cfg, Path("model.onnx"), Path("out"))


if __name__ == "__main__":
    unittest.main()
