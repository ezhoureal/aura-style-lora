# LoRA Training Workspace

This repo trains paired image-edit style-transfer LoRAs. Each training row uses a
photorealistic source image plus a target image in the desired style, and the model learns
to transfer style while preserving identity, composition, pose, silhouette, camera framing,
and important details.

The active local target is `flux2_klein_base`, a FLUX.2 Klein/Base 4B paired-edit LoRA.
Hydra config lives in `configs/local_edit_lora.yaml`; avoid ad hoc CLI argument drift by
updating that YAML when changing training or eval settings.

## Layout

- `src/lora/local_edit_common.py`: shared paths, dataset metadata, image loading, config,
  batching, and seed helpers.
- `src/lora/local_edit_flux2.py`: FLUX.2 paired-edit dataset, flow-matching objective,
  trainer, LoRA loading, and inference.
- `src/lora/local_edit_sd.py`: Stable Diffusion InstructPix2Pix dataset, trainer, UNet
  input-channel patching, and inference.
- `src/lora/local_edit_sd3.py`: Stable Diffusion 3.5 paired-edit trainer scaffold,
  transformer input-projection patching, and LoRA checkpoint plumbing. The core
  `training_step` is intentionally left as a TODO.
- `src/lora/local_edit_training.py`: Hydra training entrypoint and trainer dispatch.
- `src/lora/local_edit_inference.py`: Hydra inference entrypoint and model dispatch.
- `scripts/train_local_edit_lora.py`: local Hydra training wrapper.
- `scripts/run_local_edit_lora.py`: local Hydra inference wrapper.

## Environment

Use the checked-in `uv` environment:

```bash
uv sync --dev
```

Check the GPU before training or eval:

```bash
nvidia-smi
```

The current config is tuned for a single RTX 4090-class GPU with BF16. Training minimizes
VRAM by freezing non-LoRA modules, caching prompt embeddings, and moving the text encoder
off GPU after cache construction.

## Train

Default training uses `configs/local_edit_lora.yaml`:

```bash
uv run scripts/train_local_edit_lora.py
```

Useful reproducible overrides for a smoke run:

```bash
uv run scripts/train_local_edit_lora.py \
  training.max_train_steps=1 \
  training.checkpointing_steps=1 \
  training.output_root=outputs/smoke/flux2_local_edit_lora
```

## Evaluate

Run local batch evaluation with the final checkpoint:

```bash
uv run scripts/run_local_edit_lora.py
```

By default, `evaluation.checkpoint_dir: null` resolves to:

```text
${training.output_root}/${model_key}/checkpoint-${training.max_train_steps}

Eval outputs are written to:

```text
outputs/ablation/local_edit_eval/flux2_klein_base
```

## Notes

- `selected_models` currently contains only `flux2_klein_base`.
- SD 1.5 and SD 2.1 trainer code remains in the repo, but the active paired Flux2 path is
  the one currently configured.
- SD3.5 local paired-edit support is scaffolded in config, but its core train step is
  intentionally unimplemented.
- The eval code resizes/crops each Flux2 conditioning image to the configured eval canvas
  before inference, matching the training resolution by default and keeping VRAM predictable.
