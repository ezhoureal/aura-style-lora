# LoRA Training Workspace

This repo trains paired image-edit style-transfer LoRAs. Each training row uses a
photorealistic source image plus a target image in the desired style, and the model learns
to transfer style while preserving identity, composition, pose, silhouette, camera framing,
and important details.

The active local target is `sd35_medium`, a Stable Diffusion 3.5 Medium paired-edit LoRA.
Hydra config lives in `configs/local_edit_lora.yaml`; avoid ad hoc CLI argument drift by
updating that YAML when changing training or eval settings.

## Layout

- `src/lora/local_edit_common.py`: shared paths, dataset metadata, image loading, config,
  batching, and seed helpers.
- `src/lora/local_edit_flux2.py`: FLUX.2 paired-edit dataset, flow-matching objective,
  trainer, LoRA loading, and inference.
- `src/lora/local_edit_sd.py`: Stable Diffusion InstructPix2Pix dataset, trainer, UNet
  input-channel patching, and inference.
- `src/lora/local_edit_sd3.py`: Stable Diffusion 3.5 paired-edit dataset wiring,
  transformer input-projection patching, LoRA checkpoint plumbing, training, and inference.
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

The current config is tuned for a single RTX 4090-class GPU with BF16. SD3.5 training
minimizes VRAM by freezing non-LoRA modules, building prompt embeddings on GPU while the
transformer is still off GPU, then moving text encoders back to CPU before the training loop.

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
  training.output_root=outputs/smoke/sd3_local_edit_lora
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
outputs/ablation/local_edit_eval/sd35_medium
```

## Notes

- `selected_models` currently contains only `sd35_medium`.
- SD 1.5, SD 2.1, and FLUX.2 trainer code remains in the repo, but the active paired SD3.5
  path is the one currently configured.
- SD3.5 eval resizes/crops each conditioning image to the configured eval canvas before
  inference, matching the training resolution by default and keeping VRAM predictable.
