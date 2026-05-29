from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "flux_lora.toml"


class CommandFunc(Protocol):
    def __call__(self, args: argparse.Namespace) -> int: ...


class PipelineResult(Protocol):
    images: list[Any]


@dataclass
class PreparedItem:
    image_source: Path
    output_name: str
    prompt: str
    conditioning_source: Path | None = None
    kind: str = "label"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def parse_bool(value: bool) -> bool:
    return bool(value)


def normalize_subject(value: str) -> str:
    return value.lower().replace("-", " ").replace("_", " ").strip()


def build_prompt(subject: str, prompt_cfg: dict[str, Any]) -> str:
    return prompt_cfg["prompt_template"].format(subject=subject)


def convert_image(source: Path, target: Path) -> None:
    from PIL import Image, ImageOps
    from PIL.Image import Image as PILImage

    with Image.open(source) as image:
        transposed = cast(PILImage, ImageOps.exif_transpose(image))
        rgb_image = transposed.convert("RGB")
        rgb_image.save(target, format="PNG", optimize=True)


def prepare_dataset(args: argparse.Namespace) -> int:
    if find_spec("PIL") is None:
        print("Pillow is required. Install project dependencies first.", file=sys.stderr)
        return 1

    config = load_config(Path(args.config))
    dataset_cfg = config["dataset"]
    prompt_cfg = config["prompts"]

    source_dir = REPO_ROOT / dataset_cfg["source_dir"]
    output_dir = REPO_ROOT / dataset_cfg["prepared_dir"]
    image_dir = output_dir / "train"
    conditioning_dir = output_dir / "conditioning"
    metadata_path = image_dir / "metadata.jsonl"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    conditioning_dir.mkdir(parents=True, exist_ok=True)

    supported = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
    files = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in supported
    )
    if not files:
        print(f"No supported images found in {source_dir}", file=sys.stderr)
        return 1

    source_pattern = re.compile(r"^(?P<index>\d+)$")
    output_pattern = re.compile(r"^(?P<index>\d+)-output$")
    pairs: dict[str, dict[str, Path]] = {}
    label_images: list[Path] = []

    for path in files:
        stem = path.stem
        source_match = source_pattern.match(stem)
        output_match = output_pattern.match(stem)
        if output_match:
            pairs.setdefault(output_match.group("index"), {})["output"] = path
            continue
        if source_match:
            pairs.setdefault(source_match.group("index"), {})["source"] = path
            continue
        label_images.append(path)

    items: list[PreparedItem] = []
    pair_subject = prompt_cfg["paired_subject"]
    for pair_index in sorted(pairs, key=lambda value: int(value)):
        pair = pairs[pair_index]
        source = pair.get("source")
        output = pair.get("output")
        if source is None or output is None:
            missing = "source" if source is None else "output"
            print(f"Skipping pair {pair_index}: missing {missing} image.", file=sys.stderr)
            continue

        items.append(
            PreparedItem(
                image_source=output,
                conditioning_source=source,
                output_name=f"pair-{int(pair_index):03d}.png",
                prompt=build_prompt(pair_subject, prompt_cfg),
                kind="paired",
            )
        )

    for label_index, image in enumerate(sorted(label_images), start=1):
        items.append(
            PreparedItem(
                image_source=image,
                output_name=f"label-{label_index:03d}.png",
                prompt=build_prompt(normalize_subject(image.stem), prompt_cfg),
                kind="label",
            )
        )

    for item in items:
        target = image_dir / item.output_name
        convert_image(item.image_source, target)
        if item.conditioning_source is not None:
            convert_image(item.conditioning_source, conditioning_dir / item.output_name)

    with metadata_path.open("w", encoding="utf-8") as fh:
        for item in items:
            row = {
                "file_name": item.output_name,
                "prompt": item.prompt,
                "kind": item.kind,
            }
            if item.conditioning_source is not None:
                row["conditioning_path"] = f"../conditioning/{item.output_name}"
            fh.write(json.dumps(row) + "\n")

    summary = {
        "prepared_count": len(items),
        "paired_count": sum(1 for item in items if item.kind == "paired"),
        "label_count": sum(1 for item in items if item.kind == "label"),
        "prepared_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2))
    return 0


def run_checked(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def missing_modules(module_names: list[str]) -> list[str]:
    return [module_name for module_name in module_names if find_spec(module_name) is None]


def ensure_diffusers_checkout(cache_dir: Path, ref: str) -> Path:
    repo_dir = cache_dir / "diffusers"
    if not repo_dir.exists():
        run_checked(["git", "clone", "https://github.com/huggingface/diffusers.git", str(repo_dir)])
    run_checked(["git", "fetch", "origin"], cwd=repo_dir)
    run_checked(["git", "checkout", ref], cwd=repo_dir)
    return repo_dir


def append_flag(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(name)
        return
    command.extend([name, str(value)])


def train(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    train_cfg = config["training"]
    prompt_cfg = config["prompts"]

    missing = missing_modules(
        [
            "accelerate",
            "datasets",
            "diffusers",
            "ftfy",
            "hf_transfer",
            "peft",
            "sentencepiece",
            "tensorboard",
            "torch",
            "torchvision",
            "transformers",
        ]
    )
    if missing:
        print(
            "Training dependencies are missing: "
            + ", ".join(missing)
            + ". Run `uv sync` or `lora install` before `lora train`.",
            file=sys.stderr,
        )
        return 1

    prepared_dir = REPO_ROOT / config["dataset"]["prepared_dir"]
    if not prepared_dir.exists():
        print(
            "Prepared dataset is missing. Run `lora prepare-dataset` first.",
            file=sys.stderr,
        )
        return 1

    command = [
        "accelerate",
        "launch",
        "-m",
        train_cfg.get("training_module", "lora.trainers.flux_lowmem"),
        "--pretrained_model_name_or_path",
        train_cfg["model_name"],
        "--dataset_name",
        str(prepared_dir),
        "--caption_column",
        "prompt",
        "--instance_prompt",
        prompt_cfg["instance_prompt"],
        "--output_dir",
        str(REPO_ROOT / train_cfg["output_dir"]),
    ]

    flags = {
        "--mixed_precision": train_cfg["mixed_precision"],
        "--resolution": train_cfg["resolution"],
        "--train_batch_size": train_cfg["train_batch_size"],
        "--gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "--optimizer": train_cfg["optimizer"],
        "--learning_rate": train_cfg["learning_rate"],
        "--lr_scheduler": train_cfg["lr_scheduler"],
        "--lr_warmup_steps": train_cfg["lr_warmup_steps"],
        "--max_train_steps": train_cfg["max_train_steps"],
        "--rank": train_cfg["rank"],
        "--lora_alpha": train_cfg["lora_alpha"],
        "--validation_prompt": train_cfg.get("validation_prompt") or None,
        "--validation_epochs": train_cfg["validation_epochs"],
        "--num_validation_images": train_cfg["num_validation_images"],
        "--seed": train_cfg["seed"],
        "--report_to": train_cfg["report_to"],
        "--repeats": train_cfg["repeats"],
        "--max_sequence_length": train_cfg["max_sequence_length"],
        "--dataloader_num_workers": train_cfg.get("dataloader_num_workers"),
    }
    for flag_name, value in flags.items():
        append_flag(command, flag_name, value)

    if parse_bool(train_cfg.get("gradient_checkpointing", False)):
        command.append("--gradient_checkpointing")
    if parse_bool(train_cfg.get("cache_latents", False)):
        command.append("--cache_latents")
    if parse_bool(train_cfg.get("use_8bit_adam", False)):
        command.append("--use_8bit_adam")
    if parse_bool(train_cfg.get("push_to_hub", False)):
        command.append("--push_to_hub")

    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print("Launching training command:\n")
    print(" ".join(command))
    print()
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return 0


def infer(args: argparse.Namespace) -> int:
    try:
        import diffusers  # pyright: ignore[reportMissingImports]
        import torch
    except ImportError:
        print(
            "Inference requires diffusers and torch installed in the active environment.",
            file=sys.stderr,
        )
        return 1

    config = load_config(Path(args.config))
    infer_cfg = config["inference"]
    train_cfg = config["training"]

    lora_dir = REPO_ROOT / train_cfg["output_dir"]
    weight_name = infer_cfg["weight_name"]
    prompt = args.prompt or infer_cfg["prompt"]
    output_path = REPO_ROOT / infer_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_cls = cast(Any, getattr(diffusers, "DiffusionPipeline"))
    pipe = cast(
        Any,
        pipeline_cls.from_pretrained(
            train_cfg["model_name"],
            torch_dtype=getattr(torch, "bfloat16"),
        ),
    )
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(str(lora_dir), weight_name=weight_name)

    result = cast(
        PipelineResult,
        pipe(
            prompt=prompt,
            height=infer_cfg["height"],
            width=infer_cfg["width"],
            guidance_scale=infer_cfg["guidance_scale"],
            num_inference_steps=infer_cfg["num_inference_steps"],
            max_sequence_length=train_cfg["max_sequence_length"],
        ),
    )
    image = result.images[0]
    image.save(output_path)
    print(json.dumps({"prompt": prompt, "output_path": str(output_path)}, indent=2))
    return 0


def install(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    train_cfg = config["training"]
    cache_dir = REPO_ROOT / ".cache"
    cache_dir.mkdir(exist_ok=True)
    diffusers_dir = ensure_diffusers_checkout(cache_dir, train_cfg["diffusers_ref"])

    steps = [
        [sys.executable, "-m", "pip", "install", "-e", "."],
        [sys.executable, "-m", "pip", "install", "-e", str(diffusers_dir)],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(diffusers_dir / "examples" / "dreambooth" / "requirements_flux.txt"),
        ],
    ]
    for step in steps:
        run_checked(step, cwd=REPO_ROOT)

    note = textwrap.dedent(
        """
        Environment bootstrap complete.
        Next steps:
        1. Accept the gated model terms for black-forest-labs/FLUX.1-dev on Hugging Face.
        2. Run `hf auth login`.
        3. Run `accelerate config default`.
        4. Run `lora prepare-dataset`.
        5. Run `lora train`.
        """
    ).strip()
    print(note)
    return 0


def clean(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    paths = [
        REPO_ROOT / config["dataset"]["prepared_dir"],
        REPO_ROOT / config["training"]["output_dir"],
        REPO_ROOT / config["inference"]["output_path"],
    ]
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    print("Removed generated dataset, output weights, and sample image.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FLUX.1-dev LoRA workspace utilities.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the TOML config file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-dataset", help="Normalize images into a local HF dataset."
    )
    prepare_parser.set_defaults(func=prepare_dataset)

    install_parser = subparsers.add_parser(
        "install", help="Install local and upstream training dependencies."
    )
    install_parser.set_defaults(func=install)

    train_parser = subparsers.add_parser(
        "train", help="Launch the official diffusers FLUX LoRA trainer."
    )
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser(
        "infer", help="Run a quick inference pass with the trained LoRA."
    )
    infer_parser.add_argument("prompt", nargs="?", help="Optional prompt override.")
    infer_parser.set_defaults(func=infer)

    clean_parser = subparsers.add_parser("clean", help="Remove generated artifacts.")
    clean_parser.set_defaults(func=clean)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        func = cast(CommandFunc, args.func)
        return func(args)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1
