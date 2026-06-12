from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from lora.local_edit_training import (
    REPO_ROOT,
    expand_unet_conv_in_for_ip2p,
    load_pair_examples,
)


class TinyConfig:
    in_channels: int

    def __init__(self) -> None:
        self.in_channels = 4


class TinyUnet(nn.Module):
    conv_in: nn.Conv2d
    config: TinyConfig

    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(4, 8, 3, padding=1)
        self.config = TinyConfig()


class DatasetTests(unittest.TestCase):
    def test_loads_only_valid_paired_rows(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            conditioning_dir = root / "conditioning"
            train_dir.mkdir()
            conditioning_dir.mkdir()
            (train_dir / "pair-001.png").write_bytes(b"target")
            (conditioning_dir / "pair-001.png").write_bytes(b"source")
            rows: list[dict[str, Any]] = [
                {
                    "file_name": "pair-001.png",
                    "conditioning_path": "../conditioning/pair-001.png",
                    "prompt": "make it aura style",
                    "kind": "paired",
                },
                {
                    "file_name": "label-001.png",
                    "prompt": "label-only row",
                    "kind": "label",
                },
            ]
            with (train_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            cfg = OmegaConf.create(
                {
                    "train_dir": str(train_dir.relative_to(REPO_ROOT)),
                    "conditioning_key": "conditioning_path",
                    "image_key": "file_name",
                    "prompt_key": "prompt",
                    "kind_key": "kind",
                    "paired_kind": "paired",
                }
            )

            examples = load_pair_examples(cfg)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].prompt, "make it aura style")
        self.assertEqual(examples[0].target_path.name, "pair-001.png")
        self.assertEqual(examples[0].source_path.name, "pair-001.png")


class UnetPatchTests(unittest.TestCase):
    def test_expands_conv_in_to_accept_source_latents(self) -> None:
        unet = TinyUnet()
        original = unet.conv_in.weight.detach().clone()

        expand_unet_conv_in_for_ip2p(unet)  # type: ignore[arg-type]

        self.assertEqual(unet.conv_in.in_channels, 8)
        self.assertEqual(unet.config.in_channels, 8)
        self.assertTrue(torch.equal(unet.conv_in.weight[:, :4], original))
        self.assertTrue(
            torch.equal(unet.conv_in.weight[:, 4:], torch.zeros_like(unet.conv_in.weight[:, 4:]))
        )


if __name__ == "__main__":
    unittest.main()
