#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DIR = REPO_ROOT / "training_data" / "flux_aura_style" / "train"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "fal_flux2_edit_lora"
DEFAULT_ENDPOINT = "fal-ai/flux-2-trainer-v2/edit"
DEFAULT_CAPTION = (
    "Apply a radiant aura lighting style with smooth colorful gradients, ethereal haze, subtle "
    "contour lighting, and a refined cinematic glow while preserving the subject and composition."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package paired image-edit data and submit a FLUX.2 edit LoRA training job to fal."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Prepared train directory containing metadata.jsonl and target images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the packaged zip, result JSON, and downloaded LoRA files.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional explicit path for the generated training zip.",
    )
    parser.add_argument(
        "--default-caption",
        default=DEFAULT_CAPTION,
        help="Fallback edit instruction passed to fal if any pair has no prompt text file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Training steps. fal accepts 100 to 10000 in increments of 100.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00005,
        help="LoRA learning rate.",
    )
    parser.add_argument(
        "--output-lora-format",
        choices=("fal", "comfy"),
        default="fal",
        help="Output weight naming format.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="fal endpoint id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of paired examples to package.",
    )
    parser.add_argument(
        "--start-pair",
        type=int,
        default=None,
        help="First pair number to include, inclusive.",
    )
    parser.add_argument(
        "--end-pair",
        type=int,
        default=None,
        help="Last pair number to include, inclusive.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download result files after training completes.",
    )
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="Only create the zip; do not upload or start training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected pairs without creating a zip or calling fal.",
    )
    return parser.parse_args()


def pair_number(path: str | Path) -> int:
    match = re.fullmatch(r"pair-(\d+)\.[^.]+", Path(path).name)
    if not match:
        raise ValueError(f"Not a pair file name: {path}")
    return int(match.group(1))


def read_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def selected_pair_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    metadata_path = args.train_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    rows = []
    for row in read_metadata(metadata_path):
        if row.get("kind") != "paired":
            continue
        file_name = row.get("file_name")
        conditioning_path = row.get("conditioning_path")
        if not isinstance(file_name, str) or not isinstance(conditioning_path, str):
            continue
        number = pair_number(file_name)
        if args.start_pair is not None and number < args.start_pair:
            continue
        if args.end_pair is not None and number > args.end_pair:
            continue
        rows.append(row)

    rows.sort(key=lambda row: pair_number(row["file_name"]))
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def resolve_conditioning_path(train_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (train_dir / path).resolve()


def validate_rows(train_dir: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        target_path = train_dir / row["file_name"]
        conditioning_path = resolve_conditioning_path(train_dir, row["conditioning_path"])
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target image: {target_path}")
        if not conditioning_path.exists():
            raise FileNotFoundError(f"Missing conditioning image: {conditioning_path}")


def build_zip(train_dir: Path, rows: list[dict[str, Any]], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{zip_path.name}.",
        suffix=".tmp",
        dir=zip_path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for row in rows:
                target_path = train_dir / row["file_name"]
                conditioning_path = resolve_conditioning_path(train_dir, row["conditioning_path"])
                stem = Path(row["file_name"]).stem

                archive.write(conditioning_path, f"{stem}_start{conditioning_path.suffix.lower()}")
                archive.write(target_path, f"{stem}_end{target_path.suffix.lower()}")

                prompt = row.get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    archive.writestr(f"{stem}.txt", prompt.strip())

        tmp_path.replace(zip_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


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


def train_with_fal(args: argparse.Namespace, zip_path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        import fal_client
    except ImportError as exc:
        raise RuntimeError(
            "fal-client is not installed. Run `uv sync` after this script was added, or install "
            "it with `uv pip install fal-client`."
        ) from exc

    image_data_url = fal_client.upload_file(str(zip_path)) # type: ignore
    print(f"Uploaded dataset zip: {image_data_url}", flush=True)

    result = fal_client.subscribe(
        args.endpoint,
        arguments={
            "image_data_url": image_data_url,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "default_caption": args.default_caption,
            "output_lora_format": args.output_lora_format,
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    if hasattr(result, "data"):
        return dict(result.data), getattr(result, "request_id", None)
    return dict(result), None


def download_file(file_info: dict[str, Any], output_dir: Path) -> Path | None:
    url = file_info.get("url")
    if not isinstance(url, str) or not url:
        return None

    file_name = file_info.get("file_name")
    if not isinstance(file_name, str) or not file_name:
        file_name = Path(urllib.parse.urlparse(url).path).name or "downloaded-file"

    output_path = output_dir / file_name
    urllib.request.urlretrieve(url, output_path)
    return output_path


def main() -> int:
    args = parse_args()
    train_dir = args.train_dir.resolve()
    output_dir = args.output_dir.resolve()
    zip_path = (args.zip_path or (output_dir / "flux2-edit-lora-pairs.zip")).resolve()

    if args.steps < 100 or args.steps > 10000 or args.steps % 100 != 0:
        print("--steps must be between 100 and 10000, in increments of 100.", file=sys.stderr)
        return 1
    if not args.package_only and not args.dry_run and not os.environ.get("FAL_KEY"):
        print("Missing fal key. Set FAL_KEY before submitting training.", file=sys.stderr)
        return 1

    rows = selected_pair_rows(args)
    if not rows:
        print("No paired examples selected.", file=sys.stderr)
        return 1
    validate_rows(train_dir, rows)

    print(
        json.dumps(
            {
                "pairs": len(rows),
                "first_pair": rows[0]["file_name"],
                "last_pair": rows[-1]["file_name"],
                "zip_path": str(zip_path),
                "endpoint": args.endpoint,
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "package_only": args.package_only,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    if args.dry_run:
        for row in rows:
            print(f"{row['conditioning_path']} -> {row['file_name']}")
        return 0

    build_zip(train_dir, rows, zip_path)
    print(f"Wrote dataset zip: {zip_path} ({zip_path.stat().st_size:,} bytes)")

    if args.package_only:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    result, request_id = train_with_fal(args, zip_path)
    elapsed = time.time() - started_at

    result_path = output_dir / "fal-training-result.json"
    result_path.write_text(
        json.dumps({"request_id": request_id, "elapsed_seconds": elapsed, "result": result}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote result JSON: {result_path}")

    if not args.no_download:
        for key in ("diffusers_lora_file", "config_file"):
            value = result.get(key)
            if isinstance(value, dict):
                downloaded = download_file(value, output_dir)
                if downloaded is not None:
                    print(f"Downloaded {key}: {downloaded}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
