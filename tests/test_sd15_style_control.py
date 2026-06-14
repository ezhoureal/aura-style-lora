from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

from lora.local_edit_common import REPO_ROOT
from lora.sd15_style_control import (
    ControlRuntime,
    Stage0Example,
    discover_stage0_examples,
    make_control_image,
    prompt_for_example,
    selected_baseline_keys,
    stage0_seed,
)


class SD15StyleControlConfigTests(unittest.TestCase):
    def test_selected_baselines_preserve_config_order(self) -> None:
        cfg = OmegaConf.create({"stage0": {"selected_baselines": ["a_img2img", "c_lora_canny"]}})

        self.assertEqual(selected_baseline_keys(cfg), ["a_img2img", "c_lora_canny"])

    def test_stage0_examples_use_metadata_captions(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp)
            image_dir = root / "stage0"
            image_dir.mkdir()
            Image.new("RGB", (12, 10), "white").save(image_dir / "horse.png")
            metadata_path = root / "metadata.jsonl"
            metadata_path.write_text(
                '{"file_name": "horse.png", "caption": "a horse in a field"}\n',
                encoding="utf-8",
            )
            cfg = OmegaConf.create(
                {
                    "stage0": {
                        "input_dir": str(image_dir.relative_to(REPO_ROOT)),
                        "caption_metadata_path": str(metadata_path.relative_to(REPO_ROOT)),
                        "image_key": "file_name",
                        "caption_key": "caption",
                        "default_caption": "fallback",
                        "limit": None,
                    }
                }
            )

            examples = discover_stage0_examples(cfg)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].caption, "a horse in a field")

    def test_prompt_and_seed_are_reproducible(self) -> None:
        cfg = OmegaConf.create(
            {
                "stage0": {
                    "prompt_template": "<AURA_STYLE> style, {caption}",
                    "seed": 42,
                }
            }
        )

        prompt = prompt_for_example(
            cfg, Stage0Example(image_path=Path("mountain.png"), caption="a mountain landscape")
        )

        self.assertEqual(prompt, "<AURA_STYLE> style, a mountain landscape")
        self.assertEqual(stage0_seed(cfg), 42)


class SD15StyleControlPreprocessorTests(unittest.TestCase):
    def test_builtin_control_preprocessors_return_configured_size(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            source_path = Path(tmp) / "flower.png"
            Image.new("RGB", (40, 20), "black").save(source_path)

            for preprocessor in ["canny", "lineart", "tile"]:
                image = make_control_image(
                    ControlRuntime(
                        name=preprocessor,
                        scale=1.0,
                        start=0.0,
                        end=1.0,
                        preprocessor=preprocessor,
                        precomputed_dir=None,
                        threshold=64,
                    ),
                    source_path,
                    16,
                    12,
                )

                self.assertEqual(image.size, (16, 12))
                self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
