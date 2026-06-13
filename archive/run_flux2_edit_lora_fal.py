#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_OUTPUT_DIR = REPO_ROOT / "outputs" / "fal_flux2_edit_lora"
DEFAULT_TRAINING_RESULT = DEFAULT_TRAINING_OUTPUT_DIR / "fal-training-result.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "fal_flux2_edit_inference"
DEFAULT_ENDPOINT = "fal-ai/flux-2/lora/edit"
DEFAULT_PROMPT = (
    "Transform this photorealistic image into the trained radiant aura style: smooth colorful "
    "gradients, ethereal haze, subtle contour lighting, and a refined cinematic glow. Preserve the "
    "subject identity, composition, pose, silhouette, camera framing, and important details."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the trained FLUX.2 edit LoRA on fal to a photorealistic input image and download "
            "the stylized result."
        )
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Photorealistic input image path or URL.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Edit instruction sent to FLUX.2 LoRA Edit.",
    )
    parser.add_argument(
        "--lora",
        default=None,
        help=(
            "LoRA weights URL, Hugging Face repo ID, or local safetensors path. If omitted, the "
            "script reads --training-result or uses a local safetensors file from the training "
            "output directory."
        ),
    )
    parser.add_argument(
        "--training-result",
        type=Path,
        default=DEFAULT_TRAINING_RESULT,
        help="fal training result JSON written by scripts/train_flux2_edit_lora_fal.py.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA strength passed as the LoRAInput scale.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for downloaded images and the inference result JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path for the first downloaded output image.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="fal endpoint id.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.5,
        help="Prompt adherence. fal default is 2.5.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=28,
        help="Number of inference steps. fal accepts 4 to 50.",
    )
    parser.add_argument(
        "--image-size",
        default=None,
        help='Optional output size as WIDTHxHEIGHT, for example "1024x1024". If omitted, fal chooses.',
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate. fal accepts 1 to 4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible generations.",
    )
    parser.add_argument(
        "--acceleration",
        choices=("none", "regular", "high"),
        default="regular",
        help="fal acceleration level.",
    )
    parser.add_argument(
        "--output-format",
        choices=("jpeg", "png", "webp"),
        default="png",
        help="Output image format.",
    )
    parser.add_argument(
        "--enable-prompt-expansion",
        action="store_true",
        help="Ask fal to expand the prompt before generation.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable fal safety checker.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved request arguments without calling fal.",
    )
    return parser.parse_args()


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"}


def parse_image_size(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    try:
        width_text, height_text = value.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise ValueError('--image-size must look like "1024x1024".') from exc

    if not 512 <= width <= 2048 or not 512 <= height <= 2048:
        raise ValueError("--image-size width and height must be between 512 and 2048 pixels.")
    return {"width": width, "height": height}


def on_queue_update(update: object) -> None:
    try:
        import fal_client
    except ImportError:
        return

    if isinstance(update, fal_client.InProgress) and update.logs:
        for log in update.logs:
            message = log.get("message")
            if message:
                print(message, flush=True)


def load_training_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("result", data)
    if not isinstance(result, dict):
        raise ValueError(f"Training result JSON has no object result: {path}")
    return result


def lora_from_training_result(path: Path) -> str | None:
    result = load_training_result(path)
    if result is None:
        return None

    for key in ("diffusers_lora_file", "lora_file", "lora"):
        value = result.get(key)
        if isinstance(value, dict) and isinstance(value.get("url"), str):
            return value["url"]
        if isinstance(value, str):
            return value

    loras = result.get("loras")
    if isinstance(loras, list):
        for value in loras:
            if isinstance(value, dict) and isinstance(value.get("url"), str):
                return value["url"]
            if isinstance(value, str):
                return value
    return None


def latest_local_lora(training_output_dir: Path) -> Path | None:
    candidates = sorted(
        training_output_dir.glob("*.safetensors"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_lora(args: argparse.Namespace) -> str:
    if args.lora:
        return args.lora

    lora = lora_from_training_result(args.training_result.resolve())
    if lora:
        return lora

    local_lora = latest_local_lora(args.training_result.resolve().parent)
    if local_lora is not None:
        return str(local_lora)

    raise FileNotFoundError(
        "Could not find LoRA weights. Pass --lora, or run training without --no-download so "
        f"{args.training_result} contains a diffusers_lora_file URL."
    )


def upload_input_image(value: str, *, dry_run: bool) -> str:
    if is_url(value):
        return value

    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    if dry_run:
        return str(path)

    import fal_client

    uploaded_url = fal_client.upload_file(path)
    print(f"Uploaded {path.name}: {uploaded_url}", flush=True)
    return uploaded_url


def upload_lora_if_local(value: str, *, dry_run: bool) -> str:
    if is_url(value):
        return value

    path = Path(value).expanduser().resolve()
    if not path.exists():
        # The API also accepts Hugging Face repo IDs, which look like "owner/repo".
        return value
    if dry_run:
        return str(path)

    import fal_client

    uploaded_url = fal_client.upload_file(path)
    print(f"Uploaded {path.name}: {uploaded_url}", flush=True)
    return uploaded_url


def output_file_name(image: dict[str, Any], index: int, output_format: str) -> str:
    file_name = image.get("file_name")
    if isinstance(file_name, str) and file_name:
        return file_name

    url = image.get("url")
    if isinstance(url, str):
        parsed_name = Path(urllib.parse.urlparse(url).path).name
        if parsed_name:
            return parsed_name

    suffix = "jpg" if output_format == "jpeg" else output_format
    return f"stylized-{index + 1:02d}.{suffix}"


def download_images(
    images: list[dict[str, Any]],
    output_dir: Path,
    output_path: Path | None,
    output_format: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for index, image in enumerate(images):
        url = image.get("url")
        if not isinstance(url, str) or not url:
            continue

        if index == 0 and output_path is not None:
            destination = output_path.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination = output_dir / output_file_name(image, index, output_format)

        urllib.request.urlretrieve(url, destination)
        downloaded.append(destination)

    return downloaded


def build_arguments(
    args: argparse.Namespace,
    image_url: str,
    lora_url: str,
) -> dict[str, Any]:
    request_arguments: dict[str, Any] = {
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_images": args.num_images,
        "acceleration": args.acceleration,
        "enable_prompt_expansion": args.enable_prompt_expansion,
        "enable_safety_checker": not args.disable_safety_checker,
        "output_format": args.output_format,
        "image_urls": [image_url],
        "loras": [{"path": lora_url, "scale": args.lora_scale}],
    }

    image_size = parse_image_size(args.image_size)
    if image_size is not None:
        request_arguments["image_size"] = image_size
    if args.seed is not None:
        request_arguments["seed"] = args.seed

    return request_arguments


def run_with_fal(
    endpoint: str, request_arguments: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    import fal_client

    result = fal_client.subscribe(
        endpoint,
        arguments=request_arguments,
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    if hasattr(result, "data"):
        return dict(result.data), getattr(result, "request_id", None)
    return dict(result), None


def main() -> int:
    args = parse_args()

    if args.num_images < 1 or args.num_images > 4:
        print("--num-images must be between 1 and 4.", file=sys.stderr)
        return 1
    if args.num_inference_steps < 4 or args.num_inference_steps > 50:
        print("--num-inference-steps must be between 4 and 50.", file=sys.stderr)
        return 1
    if args.guidance_scale < 0 or args.guidance_scale > 20:
        print("--guidance-scale must be between 0 and 20.", file=sys.stderr)
        return 1
    if not args.dry_run and not os.environ.get("FAL_KEY"):
        print("Missing fal key. Set FAL_KEY before running inference.", file=sys.stderr)
        return 1

    try:
        input_image_url = upload_input_image(args.input_image, dry_run=args.dry_run)
        lora_url = upload_lora_if_local(resolve_lora(args), dry_run=args.dry_run)
        request_arguments = build_arguments(args, input_image_url, lora_url)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "endpoint": args.endpoint,
                "input_image": input_image_url,
                "lora": lora_url,
                "lora_scale": args.lora_scale,
                "output_dir": str(args.output_dir.resolve()),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    if args.dry_run:
        print(json.dumps(request_arguments, indent=2))
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    result, request_id = run_with_fal(args.endpoint, request_arguments)
    elapsed = time.time() - started_at

    images = result.get("images", [])
    if not isinstance(images, list):
        print("fal response did not contain an images list.", file=sys.stderr)
        return 1

    downloaded = download_images(
        [image for image in images if isinstance(image, dict)],
        output_dir,
        args.output_path,
        args.output_format,
    )
    result_path = output_dir / "fal-inference-result.json"
    result_path.write_text(
        json.dumps(
            {
                "request_id": request_id,
                "elapsed_seconds": elapsed,
                "arguments": request_arguments,
                "result": result,
                "downloaded_images": [str(path) for path in downloaded],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote result JSON: {result_path}")
    for path in downloaded:
        print(f"Downloaded image: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
