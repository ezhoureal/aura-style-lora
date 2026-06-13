from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from PIL import Image
from PIL.Image import Image as PILImage

from lora.local_edit_inference import (
    batched_paths,
    generators_for_batch,
    run_flux2_batch,
    run_sd_batch,
)
from lora.local_edit_training import REPO_ROOT


class FakeFlux2Pipeline:
    calls: list[dict[str, Any]]

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return FakePipelineOutput([Image.new("RGB", (8, 8), "white")])


class FakeStableDiffusionPipeline:
    calls: list[dict[str, Any]]

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        images = kwargs["image"]
        return FakePipelineOutput([Image.new("RGB", (16, 12), "white") for _image in images])


class FakePipelineOutput:
    images: list[PILImage]

    def __init__(self, images: list[PILImage]) -> None:
        self.images = images


class BatchHelperTests(unittest.TestCase):
    def test_batched_paths_chunks_inputs_without_dropping_tail(self) -> None:
        paths = [Path(f"image-{index}.png") for index in range(5)]

        chunks = batched_paths(paths, 2)

        self.assertEqual(chunks, [paths[:2], paths[2:4], paths[4:]])

    def test_batched_paths_rejects_non_positive_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "Batch size must be at least 1"):
            batched_paths([Path("image.png")], 0)

    def test_generators_for_batch_offsets_seed_per_image(self) -> None:
        generators = generators_for_batch(10, "cpu", 4, 2)

        self.assertIsInstance(generators, list)
        self.assertEqual(generators[0].initial_seed(), 14)
        self.assertEqual(generators[1].initial_seed(), 15)


class StableDiffusionBatchInferenceTests(unittest.TestCase):
    def test_sd_batch_resizes_condition_images_to_configured_eval_size(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp)
            input_paths = [root / "wide.png", root / "tall.png"]
            Image.new("RGB", (2000, 1000), "black").save(input_paths[0])
            Image.new("RGB", (600, 1200), "black").save(input_paths[1])

            cfg = OmegaConf.create(
                {
                    "evaluation": {
                        "prompt": "make it aura style",
                        "height": 640,
                        "width": 384,
                        "num_inference_steps": 8,
                        "guidance_scale": 4.0,
                        "image_guidance_scale": 1.5,
                        "seed": 100,
                    }
                }
            )
            pipe = FakeStableDiffusionPipeline()

            images = run_sd_batch(pipe, cfg, input_paths, "cpu", 6)  # type: ignore[arg-type]

        self.assertEqual(len(images), 2)
        self.assertEqual(len(pipe.calls), 1)
        self.assertEqual(pipe.calls[0]["prompt"], ["make it aura style", "make it aura style"])
        self.assertEqual([image.size for image in pipe.calls[0]["image"]], [(384, 640), (384, 640)])
        self.assertEqual(pipe.calls[0]["num_inference_steps"], 8)
        self.assertEqual(pipe.calls[0]["guidance_scale"], 4.0)
        self.assertEqual(pipe.calls[0]["image_guidance_scale"], 1.5)
        generators = pipe.calls[0]["generator"]
        self.assertIsInstance(generators, list)
        self.assertEqual(generators[0].initial_seed(), 106)
        self.assertEqual(generators[1].initial_seed(), 107)


class Flux2BatchInferenceTests(unittest.TestCase):
    def test_flux2_batch_call_uses_independent_image_calls_and_klein_encoder_layers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp)
            input_paths = [root / "a.png", root / "b.png"]
            for input_path in input_paths:
                Image.new("RGB", (12, 10), "black").save(input_path)

            cfg = OmegaConf.create(
                {
                    "evaluation": {
                        "prompt": "make it aura style",
                        "height": 640,
                        "width": 384,
                        "num_inference_steps": 8,
                        "guidance_scale": 4.0,
                        "seed": 100,
                        "max_sequence_length": 512,
                        "text_encoder_out_layers": [9, 18, 27],
                    }
                }
            )
            pipe = FakeFlux2Pipeline()

            images = run_flux2_batch(pipe, cfg, input_paths, "cpu", 6)

        self.assertEqual(len(images), 2)
        self.assertTrue(all(isinstance(image, PILImage) for image in images))
        self.assertEqual(len(pipe.calls), 2)
        self.assertEqual(pipe.calls[0]["prompt"], "make it aura style")
        self.assertEqual(pipe.calls[1]["prompt"], "make it aura style")
        self.assertIsInstance(pipe.calls[0]["image"], PILImage)
        self.assertIsInstance(pipe.calls[1]["image"], PILImage)
        self.assertEqual(pipe.calls[0]["image"].size, (384, 640))
        self.assertEqual(pipe.calls[1]["image"].size, (384, 640))
        self.assertEqual(pipe.calls[0]["height"], 640)
        self.assertEqual(pipe.calls[0]["width"], 384)
        self.assertEqual(pipe.calls[0]["text_encoder_out_layers"], (9, 18, 27))
        self.assertEqual(pipe.calls[0]["generator"].initial_seed(), 106)
        self.assertEqual(pipe.calls[1]["generator"].initial_seed(), 107)


if __name__ == "__main__":
    unittest.main()
