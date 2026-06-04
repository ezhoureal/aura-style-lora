#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LORA = REPO_ROOT / "fal_flux2_edit_lora" / "pytorch_lora_weights.safetensors"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "local_flux2_edit_inference"
DEFAULT_MODEL = "diffusers/FLUX.2-dev-bnb-4bit"
DEFAULT_PROMPT = (
    "Transform this photorealistic image into the trained radiant aura style: smooth colorful "
    "gradients, ethereal haze, subtle contour lighting, and a refined cinematic glow. Preserve the "
    "subject identity, composition, pose, silhouette, camera framing, and important details."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local FLUX.2 image editing with the fal-trained LoRA, or validate it offline."
    )
    parser.add_argument("input_image", nargs="?", type=Path, help="Photorealistic input image.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Edit prompt.")
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA, help="Input LoRA safetensors file.")
    parser.add_argument(
        "--converted-lora",
        type=Path,
        default=None,
        help="Optional path for a converted diffusers-format LoRA safetensors file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local path or Hugging Face model id. Defaults to the 4-bit FLUX.2-dev diffusers repo.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Pipeline dtype. Use bfloat16 on modern NVIDIA GPUs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help='Optional diffusers/accelerate device map, for example "balanced".',
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download model files from Hugging Face.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate/convert LoRA against the default Flux2Transformer2DModel shape without loading the base model.",
    )
    return parser.parse_args()


def dtype_from_arg(value: str) -> torch.dtype | str:
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def require_module(import_name: str, install_name: str | None = None) -> None:
    if importlib.util.find_spec(import_name) is None:
        package = install_name or import_name
        raise RuntimeError(f"Missing required package `{package}`. Install it with `uv add {package}`.")


def uses_4bit_model(model: str) -> bool:
    return "bnb-4bit" in model.lower() or "4bit" in model.lower()


def preflight_environment(args: argparse.Namespace) -> None:
    if args.check_only:
        return

    require_module("google.protobuf", "protobuf")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if uses_4bit_model(args.model):
        require_module("bitsandbytes")
        if device_name == "cpu" or not torch.cuda.is_available():
            raise RuntimeError(
                "The 4-bit FLUX.2 model needs a CUDA GPU with bitsandbytes. "
                "This environment does not expose CUDA to PyTorch."
            )
def convert_fal_key(key: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    prefix = "base_model.model."
    if not key.startswith(prefix):
        return {key: tensor}

    body = key.removeprefix(prefix)
    suffix = ".lora_A.weight" if body.endswith(".lora_A.weight") else ".lora_B.weight"
    base = body.removesuffix(suffix)

    simple_map = {
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
        "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "final_layer.linear": "proj_out",
    }
    if base in simple_map:
        return {f"transformer.{simple_map[base]}{suffix}": tensor}

    double_match = re.fullmatch(r"double_blocks\.(\d+)\.(img_attn|txt_attn)\.(qkv|proj)", base)
    if double_match:
        block, stream, layer = double_match.groups()
        stem = f"transformer.transformer_blocks.{block}.attn"
        if layer == "proj":
            target = "to_out.0" if stream == "img_attn" else "to_add_out"
            return {f"{stem}.{target}{suffix}": tensor}

        targets = (
            ("to_q", "to_k", "to_v")
            if stream == "img_attn"
            else ("add_q_proj", "add_k_proj", "add_v_proj")
        )
        if suffix == ".lora_A.weight":
            return {f"{stem}.{target}{suffix}": tensor.clone() for target in targets}

        chunks = tensor.chunk(3, dim=0)
        return {f"{stem}.{target}{suffix}": chunk.contiguous() for target, chunk in zip(targets, chunks)}

    single_match = re.fullmatch(r"single_blocks\.(\d+)\.(linear1|linear2)", base)
    if single_match:
        block, layer = single_match.groups()
        target = "to_qkv_mlp_proj" if layer == "linear1" else "to_out"
        return {f"transformer.single_transformer_blocks.{block}.attn.{target}{suffix}": tensor}

    raise ValueError(f"Unsupported fal LoRA key: {key}")


def convert_fal_lora_to_diffusers(input_path: Path, output_path: Path) -> dict[str, Any]:
    state = load_file(input_path)
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        for new_key, new_tensor in convert_fal_key(key, tensor).items():
            if new_key in converted:
                raise ValueError(f"Duplicate converted LoRA key: {new_key}")
            converted[new_key] = new_tensor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, output_path, metadata={"format": "pt"})
    return {
        "input_keys": len(state),
        "converted_keys": len(converted),
        "input_bytes": input_path.stat().st_size,
        "converted_bytes": output_path.stat().st_size,
    }


def expected_linear_shapes() -> dict[str, tuple[int, ...]]:
    from accelerate import init_empty_weights
    from diffusers import Flux2Transformer2DModel

    with init_empty_weights():
        model = Flux2Transformer2DModel()
    return {
        f"transformer.{name}": tuple(module.weight.shape)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Linear"
    }


def validate_converted_lora(path: Path) -> dict[str, Any]:
    state = load_file(path)
    shapes = expected_linear_shapes()
    missing_targets = []
    bad_shapes = []
    ranks = set()

    for key, tensor in state.items():
        if key.endswith(".lora_A.weight"):
            target = key.removesuffix(".lora_A.weight")
            ranks.add(tensor.shape[0])
            expected = shapes.get(target)
            if expected is None:
                missing_targets.append(target)
            elif tuple(tensor.shape[1:]) != (expected[1],):
                bad_shapes.append((key, tuple(tensor.shape), expected))
        elif key.endswith(".lora_B.weight"):
            target = key.removesuffix(".lora_B.weight")
            ranks.add(tensor.shape[1])
            expected = shapes.get(target)
            if expected is None:
                missing_targets.append(target)
            elif tuple(tensor.shape[:1]) != (expected[0],):
                bad_shapes.append((key, tuple(tensor.shape), expected))
        else:
            missing_targets.append(key)

    return {
        "keys": len(state),
        "target_modules": len({key.rsplit(".lora_", 1)[0] for key in state}),
        "ranks": sorted(ranks),
        "missing_targets": sorted(set(missing_targets)),
        "bad_shapes": bad_shapes,
        "valid": not missing_targets and not bad_shapes,
    }


def load_flux2_pipeline(args: argparse.Namespace, dtype: torch.dtype | str, device_name: str):
    from diffusers import Flux2Pipeline

    if uses_4bit_model(args.model) and device_name.startswith("cuda"):
        from diffusers import AutoModel
        from transformers import Mistral3ForConditionalGeneration

        print("Loading 4-bit FLUX.2 with local text encoder on CPU and model CPU offload.", flush=True)
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            args.model,
            subfolder="text_encoder",
            torch_dtype=dtype,
            device_map="cpu",
            local_files_only=args.local_files_only,
        )
        transformer = AutoModel.from_pretrained(
            args.model,
            subfolder="transformer",
            torch_dtype=dtype,
            device_map="cpu",
            local_files_only=args.local_files_only,
        )
        pipe = Flux2Pipeline.from_pretrained(
            args.model,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        )
        pipe.enable_model_cpu_offload()
        return pipe

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    elif device_name.startswith("cuda"):
        load_kwargs["device_map"] = device_name

    pipe = Flux2Pipeline.from_pretrained(args.model, **load_kwargs)
    if "device_map" not in load_kwargs:
        pipe.to(device_name)
    return pipe


def run_inference(args: argparse.Namespace, lora_path: Path) -> Path:
    from diffusers.utils import load_image

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_arg(args.torch_dtype)

    print("Using text encoder mode: local", flush=True)
    pipe = load_flux2_pipeline(args, dtype, device_name)
    pipe.load_lora_weights(str(lora_path), adapter_name="aura")
    pipe.set_adapters(["aura"], adapter_weights=[args.lora_scale])

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device_name).manual_seed(args.seed)

    input_image = load_image(str(args.input_image))
    call_kwargs: dict[str, Any] = {
        "image": input_image,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    call_kwargs["prompt"] = args.prompt

    image = pipe(**call_kwargs).images[0]
    output_path = args.output_path or (args.output_dir / "flux2-local-stylized.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def main() -> int:
    args = parse_args()
    lora_path = args.lora.expanduser().resolve()
    if not lora_path.exists():
        print(f"LoRA file does not exist: {lora_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converted_path = (
        args.converted_lora.expanduser().resolve()
        if args.converted_lora
        else output_dir / "pytorch_lora_weights.diffusers.safetensors"
    )

    try:
        conversion = convert_fal_lora_to_diffusers(lora_path, converted_path)
        validation = validate_converted_lora(converted_path)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    report: dict[str, Any] = {
        "model": args.model,
        "text_encoder_mode": "local",
        "lora": str(lora_path),
        "converted_lora": str(converted_path),
        "conversion": conversion,
        "validation": validation,
    }
    print(json.dumps(report, indent=2, default=str))

    if not validation["valid"]:
        print("Converted LoRA did not validate against Flux2Transformer2DModel.", file=sys.stderr)
        return 1
    if args.check_only:
        return 0
    if args.input_image is None:
        print("input_image is required unless --check-only is set.", file=sys.stderr)
        return 1
    try:
        preflight_environment(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    started_at = time.time()
    try:
        image_path = run_inference(args, converted_path)
    except Exception as exc:
        print(f"Local inference failed after {time.time() - started_at:.1f}s: {exc}", file=sys.stderr)
        return 1

    print(f"Saved local FLUX.2 edit output: {image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
