from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.mapping import inject_adapter_in_model
from safetensors.torch import save_file
from torch import nn

from lora.local_edit_training import (
    apply_lora_checkpoint,
    Flux2PairedEditLoraTrainer,
    REPO_ROOT,
    StableDiffusion3PairedEditLoraTrainer,
    expand_unet_conv_in_for_ip2p,
    expand_sd3_transformer_input_for_paired_edit,
    flow_match_noisy_latents,
    flow_match_training_target,
    load_pair_examples,
    make_training_progress,
    make_trainer,
    resolve_resume_checkpoint,
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


class TinySD3PosEmbed(nn.Module):
    proj: nn.Conv2d

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(16, 8, 2, stride=2)


class TinySD3Transformer(nn.Module):
    pos_embed: TinySD3PosEmbed
    config: TinyConfig

    def __init__(self) -> None:
        super().__init__()
        self.pos_embed = TinySD3PosEmbed()
        self.config = TinyConfig()
        self.config.in_channels = 16


class TinyLoraTarget(nn.Module):
    to_q: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(2, 2, bias=False)


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


class SD3PatchTests(unittest.TestCase):
    def test_expands_input_projection_to_accept_source_latents(self) -> None:
        transformer = TinySD3Transformer()
        original = transformer.pos_embed.proj.weight.detach().clone()

        expand_sd3_transformer_input_for_paired_edit(transformer)

        self.assertEqual(transformer.pos_embed.proj.in_channels, 32)
        self.assertEqual(transformer.config.in_channels, 32)
        self.assertTrue(torch.equal(transformer.pos_embed.proj.weight[:, :16], original))
        self.assertTrue(
            torch.equal(
                transformer.pos_embed.proj.weight[:, 16:],
                torch.zeros_like(transformer.pos_embed.proj.weight[:, 16:]),
            )
        )


class FlowMatchTests(unittest.TestCase):
    def test_noisy_latents_and_training_target_match_flux_objective(self) -> None:
        clean = torch.tensor([[[1.0, 2.0]]])
        noise = torch.tensor([[[3.0, 6.0]]])
        sigmas = torch.tensor([0.25])

        noisy = flow_match_noisy_latents(clean, noise, sigmas)
        target = flow_match_training_target(clean, noise)

        self.assertTrue(torch.equal(noisy, torch.tensor([[[1.5, 3.0]]])))
        self.assertTrue(torch.equal(target, torch.tensor([[[2.0, 4.0]]])))


class ProgressTests(unittest.TestCase):
    def test_progress_bar_is_disabled_off_local_main_process(self) -> None:
        class FakeAccelerator:
            is_local_main_process = False

        progress = make_training_progress(FakeAccelerator(), 3, 0, "Training test")
        try:
            self.assertTrue(progress.disable)
            self.assertEqual(progress.total, 3)
        finally:
            progress.close()


class ResumeCheckpointTests(unittest.TestCase):
    def test_latest_checkpoint_uses_highest_manifest_step(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            output_dir = Path(tmp)
            low = output_dir / "checkpoint-000010"
            high = output_dir / "checkpoint-000020"
            low.mkdir()
            high.mkdir()
            (low / "training_manifest.json").write_text('{"step": 10}', encoding="utf-8")
            (high / "training_manifest.json").write_text('{"step": 25}', encoding="utf-8")
            cfg = OmegaConf.create({"resume_from_checkpoint": "latest"})

            checkpoint = resolve_resume_checkpoint(cfg, output_dir)

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint.step, 25)
        self.assertEqual(checkpoint.path.name, "checkpoint-000020")

    def test_configured_checkpoint_path_is_repo_relative(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint-000007"
            checkpoint_dir.mkdir()
            cfg = OmegaConf.create(
                {"resume_from_checkpoint": str(checkpoint_dir.relative_to(REPO_ROOT))}
            )

            checkpoint = resolve_resume_checkpoint(cfg, Path(tmp))

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint.step, 7)
        self.assertEqual(checkpoint.path.name, "checkpoint-000007")

    def test_lora_checkpoint_loads_saved_adapter_weights(self) -> None:
        model = TinyLoraTarget()
        inject_adapter_in_model(
            LoraConfig(r=1, lora_alpha=1, target_modules=["to_q"], init_lora_weights="gaussian"),
            model,
            adapter_name="default",
        )
        checkpoint_state = {
            "to_q.lora_A.weight": torch.full((1, 2), 0.25),
            "to_q.lora_B.weight": torch.full((2, 1), 0.5),
        }

        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint-000003"
            checkpoint_dir.mkdir()
            save_file(checkpoint_state, checkpoint_dir / "pytorch_lora_weights.safetensors")

            apply_lora_checkpoint(model, checkpoint_dir)

        state = model.state_dict()
        self.assertTrue(
            torch.equal(state["to_q.lora_A.default.weight"], checkpoint_state["to_q.lora_A.weight"])
        )
        self.assertTrue(
            torch.equal(state["to_q.lora_B.default.weight"], checkpoint_state["to_q.lora_B.weight"])
        )


class TrainerSelectionTests(unittest.TestCase):
    def test_flux2_trainer_is_selected_from_config(self) -> None:
        cfg = OmegaConf.create(
            {
                "training": {"output_root": "outputs/test"},
                "models": {
                    "flux2": {
                        "trainer": "flux2_paired_edit_lora",
                        "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-klein-base-4B",
                    }
                },
            }
        )

        trainer = make_trainer(cfg, "flux2")

        self.assertIsInstance(trainer, Flux2PairedEditLoraTrainer)

    def test_sd3_trainer_is_selected_from_config(self) -> None:
        cfg = OmegaConf.create(
            {
                "training": {"output_root": "outputs/test"},
                "models": {
                    "sd35": {
                        "trainer": "stable_diffusion_3_paired_edit_lora",
                        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-medium",
                    }
                },
            }
        )

        trainer = make_trainer(cfg, "sd35")

        self.assertIsInstance(trainer, StableDiffusion3PairedEditLoraTrainer)


if __name__ == "__main__":
    unittest.main()
