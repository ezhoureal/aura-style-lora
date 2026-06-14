#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from typing import cast

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
PROXY_ENV_VARS = (
    "LD_PRELOAD",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclass(frozen=True)
class WeightSpec:
    name: str
    repo_id: str
    local_dir: Path
    allow_patterns: tuple[str, ...]


def clear_proxy_environment() -> None:
    for key in PROXY_ENV_VARS:
        os.environ.pop(key, None)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def required_weights(root: Path) -> list[WeightSpec]:
    model_root = root / ".hf_models"
    return [
        WeightSpec(
            name="Realistic Vision SD1.5 base",
            repo_id="SG161222/Realistic_Vision_V6.0_B1_noVAE",
            local_dir=model_root / "realistic-vision-v60-b1-novae",
            allow_patterns=("*",),
        ),
        WeightSpec(
            name="SD VAE ft-mse",
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=model_root / "sd-vae-ft-mse",
            allow_patterns=("*",),
        ),
        WeightSpec(
            name="ControlNet Canny SD1.5",
            repo_id="lllyasviel/control_v11p_sd15_canny",
            local_dir=model_root / "control_v11p_sd15_canny",
            allow_patterns=("*",),
        ),
        WeightSpec(
            name="ControlNet Lineart SD1.5",
            repo_id="lllyasviel/control_v11p_sd15_lineart",
            local_dir=model_root / "control_v11p_sd15_lineart",
            allow_patterns=("*",),
        ),
        WeightSpec(
            name="ControlNet Depth SD1.5",
            repo_id="lllyasviel/control_v11f1p_sd15_depth",
            local_dir=model_root / "control_v11f1p_sd15_depth",
            allow_patterns=("*",),
        ),
        WeightSpec(
            name="ControlNet Tile SD1.5",
            repo_id="lllyasviel/control_v11f1e_sd15_tile",
            local_dir=model_root / "control_v11f1e_sd15_tile",
            allow_patterns=("*",),
        ),
        WeightSpec(
            name="IP-Adapter SD1.5",
            repo_id="h94/IP-Adapter",
            local_dir=model_root / "ip-adapter",
            allow_patterns=(
                "models/ip-adapter_sd15.bin",
                "models/ip-adapter-plus_sd15.bin",
                "image_encoder/*",
                "README.md",
            ),
        ),
    ]


def selected_weights(specs: list[WeightSpec], names: set[str]) -> list[WeightSpec]:
    if not names:
        return specs
    selected: list[WeightSpec] = []
    for spec in specs:
        if spec.local_dir.name in names or spec.repo_id in names:
            selected.append(spec)
    missing = (
        names - {spec.local_dir.name for spec in selected} - {spec.repo_id for spec in selected}
    )
    if missing:
        choices = ", ".join(spec.local_dir.name for spec in specs)
        raise ValueError(f"Unknown weight selection {sorted(missing)}. Choices: {choices}")
    return selected


def download_weight(spec: WeightSpec) -> Path:
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {spec.name}: {spec.repo_id} -> {spec.local_dir}")
    downloaded_path = cast(
        str,
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=spec.local_dir,
            allow_patterns=list(spec.allow_patterns),
            endpoint="https://hf-mirror.com",
        ),
    )
    return Path(downloaded_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SD1.5 style-control workflow weights into .hf_models."
    )
    parser.add_argument(
        "--only",
        action="append",
        help="Download one repo or local directory name. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clear_proxy_environment()
    requested = set(args.only) if args.only is not None else set()
    specs = selected_weights(required_weights(REPO_ROOT), requested)
    for spec in specs:
        path = download_weight(spec)
        print(f"Ready: {path}")


if __name__ == "__main__":
    main()
