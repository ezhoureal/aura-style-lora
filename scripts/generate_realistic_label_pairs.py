#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DIR = REPO_ROOT / "training_data" / "flux_aura_style" / "train"
DEFAULT_CONDITIONING_DIR = REPO_ROOT / "training_data" / "flux_aura_style" / "conditioning"
DEFAULT_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
DEFAULT_GENERATION_PROMPT = (
    "Use the reference image only for composition, subject identity, pose, silhouette, and camera "
    "framing. Generate a realistic natural photo of the same subject before the aura lighting style "
    "was applied. Remove colorful glow, gradients, ethereal haze, rim-light effects, painterly "
    "stylization, and text. Keep the result clean, plausible, detailed, and photorealistic."
)


@dataclass(frozen=True)
class LabelRecord:
    label_path: Path
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate realistic conditioning images for label-*.png targets and append them as "
            "new paired training records."
        )
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ARK_API_KEY"),
        help="Volcengine Ark API key. Defaults to ARK_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ARK_IMAGE_MODEL", "doubao-seedream-5-0-260128"),
        help="Ark image generation model or endpoint ID. Defaults to ARK_IMAGE_MODEL.",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Image generation API URL.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Directory containing label-*.png and metadata.jsonl.",
    )
    parser.add_argument(
        "--conditioning-dir",
        type=Path,
        default=DEFAULT_CONDITIONING_DIR,
        help="Directory where generated realistic pair conditioning images are written.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_GENERATION_PROMPT,
        help="Prompt used to generate the realistic conditioning image from each label reference.",
    )
    parser.add_argument(
        "--size",
        default="2048x2048",
        help="Requested output size. Seedream 5.0 lite accepts values such as 2048x2048.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of labels to process.",
    )
    parser.add_argument(
        "--start-label",
        type=int,
        default=None,
        help="First label number to process, inclusive.",
    )
    parser.add_argument(
        "--end-label",
        type=int,
        default=None,
        help="Last label number to process, inclusive.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between successful API calls.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries after a failed API call.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned label to pair mapping without calling the API or writing files.",
    )
    return parser.parse_args()


def label_number(path: Path) -> int:
    match = re.fullmatch(r"label-(\d+)\.png", path.name)
    if not match:
        raise ValueError(f"Not a label file: {path}")
    return int(match.group(1))


def pair_number(path: Path) -> int:
    match = re.fullmatch(r"pair-(\d+)\.png", path.name)
    if not match:
        raise ValueError(f"Not a pair file: {path}")
    return int(match.group(1))


def read_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def collect_labels(train_dir: Path, metadata_rows: list[dict[str, Any]]) -> list[LabelRecord]:
    prompts_by_name = {
        row["file_name"]: row.get("prompt", "")
        for row in metadata_rows
        if row.get("kind") == "label" and isinstance(row.get("file_name"), str)
    }
    records = []
    for label_path in sorted(train_dir.glob("label-*.png"), key=label_number):
        prompt = prompts_by_name.get(label_path.name)
        if prompt is None:
            print(f"Skipping {label_path.name}: no label metadata row found.", file=sys.stderr)
            continue
        records.append(LabelRecord(label_path=label_path, prompt=prompt))
    return records


def filter_labels(
    records: list[LabelRecord],
    start_label: int | None,
    end_label: int | None,
    limit: int | None,
) -> list[LabelRecord]:
    filtered = []
    for record in records:
        number = label_number(record.label_path)
        if start_label is not None and number < start_label:
            continue
        if end_label is not None and number > end_label:
            continue
        filtered.append(record)
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def next_pair_numbers(train_dir: Path, count: int) -> list[int]:
    existing = [pair_number(path) for path in train_dir.glob("pair-*.png")]
    start = max(existing, default=0) + 1
    return list(range(start, start + count))


def image_as_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def strip_data_url_prefix(value: str) -> str:
    if "," in value and value.lower().startswith("data:"):
        return value.split(",", 1)[1]
    return value


def ark_generate_image(args: argparse.Namespace, label_path: Path) -> bytes:
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "image": image_as_data_url(label_path),
        "size": args.size,
        "sequential_image_generation": "disabled",
        "response_format": "b64_json",
        "output_format": "png",
        "watermark": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.api_url,
        data=body,
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            data = result.get("data")
            if not data:
                raise RuntimeError(f"API response did not contain data: {result}")
            b64_json = data[0].get("b64_json")
            if not b64_json:
                raise RuntimeError(f"API response did not contain b64_json: {result}")
            return base64.b64decode(strip_data_url_prefix(b64_json))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc

        if attempt < args.retries:
            time.sleep(2**attempt)

    assert last_error is not None
    raise last_error


def append_metadata(metadata_path: Path, row: dict[str, Any]) -> None:
    with metadata_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    train_dir = args.train_dir.resolve()
    conditioning_dir = args.conditioning_dir.resolve()
    metadata_path = train_dir / "metadata.jsonl"

    if not train_dir.exists():
        print(f"Train directory does not exist: {train_dir}", file=sys.stderr)
        return 1
    if not metadata_path.exists():
        print(f"Metadata file does not exist: {metadata_path}", file=sys.stderr)
        return 1
    if not args.dry_run and not args.api_key:
        print("Missing API key. Set ARK_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    rows = read_metadata(metadata_path)
    labels = filter_labels(
        collect_labels(train_dir, rows),
        start_label=args.start_label,
        end_label=args.end_label,
        limit=args.limit,
    )
    if not labels:
        print("No label images selected.", file=sys.stderr)
        return 1

    pair_numbers = next_pair_numbers(train_dir, len(labels))
    plan = list(zip(labels, pair_numbers, strict=True))

    print(
        json.dumps(
            {
                "selected_labels": len(labels),
                "first_pair": f"pair-{pair_numbers[0]:03d}.png",
                "last_pair": f"pair-{pair_numbers[-1]:03d}.png",
                "model": args.model,
                "size": args.size,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    if args.dry_run:
        for label, number in plan:
            print(f"{label.label_path.name} -> pair-{number:03d}.png")
        return 0

    conditioning_dir.mkdir(parents=True, exist_ok=True)
    for index, (label, number) in enumerate(plan, start=1):
        pair_name = f"pair-{number:03d}.png"
        pair_target = train_dir / pair_name
        conditioning_target = conditioning_dir / pair_name
        if pair_target.exists() or conditioning_target.exists():
            print(f"Refusing to overwrite existing files for {pair_name}", file=sys.stderr)
            return 1

        print(
            f"[{index}/{len(plan)}] Generating realistic conditioning for {label.label_path.name}"
        )
        image_bytes = ark_generate_image(args, label.label_path)
        conditioning_target.write_bytes(image_bytes)
        shutil.copy2(label.label_path, pair_target)
        append_metadata(
            metadata_path,
            {
                "file_name": pair_name,
                "prompt": label.prompt,
                "kind": "paired",
                "conditioning_path": f"../conditioning/{pair_name}",
                "source_label": label.label_path.name,
            },
        )
        if args.sleep:
            time.sleep(args.sleep)

    print(f"Generated {len(plan)} realistic label pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
