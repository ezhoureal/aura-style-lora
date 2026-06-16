from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file, save_file
import torch

from lora.local_edit_common import REPO_ROOT
from lora.sd15_style_lora_training import (
    fallback_prompt,
    load_style_examples,
    style_prompt,
    write_pipeline_lora_weights,
)


class SD15StyleLoraPromptTests(unittest.TestCase):
    def test_style_prompt_replaces_plain_aura_style_with_trigger_token(self) -> None:
        cfg = OmegaConf.create({"prompt": {"trigger": "<AURA_STYLE>"}})

        prompt = style_prompt(cfg, "Render the subject in aura style with luminous edges")

        self.assertEqual(prompt, "Render the subject in <AURA_STYLE> style with luminous edges")

    def test_style_prompt_preserves_existing_trigger(self) -> None:
        cfg = OmegaConf.create({"prompt": {"trigger": "<AURA_STYLE>"}})

        prompt = style_prompt(cfg, "Render the subject in <AURA_STYLE> style")

        self.assertEqual(prompt, "Render the subject in <AURA_STYLE> style")

    def test_fallback_prompt_uses_configured_trigger_and_description(self) -> None:
        cfg = OmegaConf.create(
            {
                "prompt": {
                    "trigger": "<AURA_STYLE>",
                    "style_description": "luminous contour lighting",
                    "fallback_template": "Render the subject in {trigger} style: {style_description}",
                }
            }
        )

        self.assertEqual(
            fallback_prompt(cfg),
            "Render the subject in <AURA_STYLE> style: luminous contour lighting",
        )


class SD15StyleLoraDatasetTests(unittest.TestCase):
    def test_load_style_examples_uses_target_images_and_paired_rows(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            train_dir.mkdir()
            Image.new("RGB", (8, 8), "white").save(train_dir / "pair-001.png")
            Image.new("RGB", (8, 8), "white").save(train_dir / "label-001.png")
            rows = [
                {
                    "file_name": "pair-001.png",
                    "prompt": "Render the subject in aura style",
                    "kind": "paired",
                },
                {
                    "file_name": "label-001.png",
                    "prompt": "Label-only row",
                    "kind": "label",
                },
            ]
            with (train_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            cfg = OmegaConf.create(
                {
                    "dataset": {
                        "train_dir": str(train_dir.relative_to(REPO_ROOT)),
                        "image_key": "file_name",
                        "prompt_key": "prompt",
                        "kind_key": "kind",
                        "paired_kind": "paired",
                    },
                    "prompt": {
                        "trigger": "<AURA_STYLE>",
                        "style_description": "luminous contour lighting",
                        "fallback_template": (
                            "Render the subject in {trigger} style: {style_description}"
                        ),
                    },
                }
            )

            examples = load_style_examples(cfg)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].image_path.name, "pair-001.png")
        self.assertEqual(examples[0].prompt, "Render the subject in <AURA_STYLE> style")


class SD15StyleLoraExportTests(unittest.TestCase):
    def test_pipeline_lora_export_keeps_lora_tensors_with_unet_prefix(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp)
            save_file(
                {
                    "block.to_q.lora_A.weight": torch.ones(1),
                    "block.to_q.base_layer.weight": torch.zeros(1),
                },
                root / "pytorch_lora_weights.safetensors",
            )

            output_path = write_pipeline_lora_weights(root)
            exported = load_file(output_path)

        self.assertEqual(list(exported.keys()), ["unet.block.to_q.lora_A.weight"])


if __name__ == "__main__":
    unittest.main()
