from __future__ import annotations

# pyright: reportPrivateImportUsage=false

import sys
import unittest
from pathlib import Path

import torch
from torch import device, equal, ones, randn, zeros

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora.trainers.flux_lowmem import get_cached_latent_dist, get_cached_text_embeddings


class GetCachedTextEmbeddingsTests(unittest.TestCase):
    def test_reuses_single_text_ids_tensor_for_batched_prompts(self) -> None:
        seq_len = 5
        prompt_embed_cache = {
            "a": (
                randn(1, seq_len, 4),
                randn(1, 6),
                zeros(seq_len, 3),
            ),
            "b": (
                randn(1, seq_len, 4),
                randn(1, 6),
                ones(seq_len, 3),
            ),
        }

        prompt_embeds, pooled_prompt_embeds, text_ids = get_cached_text_embeddings(
            prompt_embed_cache,
            ["a", "b"],
            device("cpu"),
        )

        self.assertEqual(prompt_embeds.shape, (2, seq_len, 4))
        self.assertEqual(pooled_prompt_embeds.shape, (2, 6))
        self.assertEqual(text_ids.shape, (seq_len, 3))
        self.assertTrue(equal(text_ids, prompt_embed_cache["a"][2]))


class GetCachedLatentDistTests(unittest.TestCase):
    def test_rebuilds_batched_distribution_from_instance_indices(self) -> None:
        latents_cache = [
            randn(32, 64, 64),
            randn(32, 64, 64),
            randn(32, 64, 64),
        ]

        latent_dist = get_cached_latent_dist(
            latents_cache,
            [2, 0],
            device("cpu"),
            torch.float32,
        )

        self.assertEqual(latent_dist.parameters.shape, (2, 32, 64, 64))
        self.assertTrue(equal(latent_dist.parameters[0], latents_cache[2]))
        self.assertTrue(equal(latent_dist.parameters[1], latents_cache[0]))
        self.assertEqual(latent_dist.sample().shape, (2, 16, 64, 64))


if __name__ == "__main__":
    unittest.main()
