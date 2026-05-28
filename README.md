# FLUX.1-dev LoRA Training Workspace

This repo is a minimal local workspace for preparing a custom image dataset, launching a `FLUX.1-dev` LoRA run, and testing the resulting adapter.

It uses a tracked local copy of Hugging Face's `diffusers` Flux LoRA trainer, patched for lower memory use, while keeping the project-specific logic here in this repo:

- dataset normalization into a local `imagefolder` dataset
- prompt and caption templating
- reproducible training settings in `configs/flux_lora.toml`
- a simple inference command for post-train validation

## Project layout

- [configs/flux_lora.toml](/home/zireael/lora/configs/flux_lora.toml)
- [src/lora/cli.py](/home/zireael/lora/src/lora/cli.py)
- [scripts/train_dreambooth_lora_flux_lowmem.py](/home/zireael/lora/scripts/train_dreambooth_lora_flux_lowmem.py)
- [dataset](/home/zireael/lora/dataset/)

## Why this setup

Your dataset has two kinds of examples:

- numbered source/output pairs like `1.jpg -> 1-output.png`
- standalone keyword images that do not have matching source inputs

`prepare-dataset` now preserves that split. Paired records include both the target image and its matching source image, while standalone keyword images are prepared as target-only samples.

## 1. Prerequisites

You need:

- Python 3.11+
- a CUDA-capable GPU
- a Hugging Face account
- accepted access to `black-forest-labs/FLUX.1-dev`

Install and bootstrap:

```bash
uv sync
uv pip install -e .
source .venv/bin/activate
lora install
```

Then authenticate and configure Accelerate:

```bash
hf auth login
accelerate config default
```

## 2. Prepare the dataset

This will:

- convert all supported images in `dataset/` to RGB PNGs
- write target images into `training_data/flux_aura_style/train/`
- write paired source inputs into `training_data/flux_aura_style/conditioning/`
- create `training_data/flux_aura_style/train/metadata.jsonl`

Run:

```bash
lora prepare-dataset
```

The prompt template now defaults to:

```text
A strong colorful light is illuminating {subject}'s silhouette. Medium shot. The image is rendered in a smooth gradient of luminous light, giving a radiant and ethereal appearance. Subtle contour lighting highlights delicate outlines, adding depth and a refined, high-end cinematic glow
```

For numbered pairs, `{subject}` comes from `paired_subject` in [configs/flux_lora.toml](/home/zireael/lora/configs/flux_lora.toml). For standalone images, `{subject}` is derived from the filename stem.

## 3. Share or fetch the dataset

Uploading the prepared dataset to Hugging Face is recommended if other users should reproduce the run. Upload `training_data/flux_aura_style`, not the raw `dataset/` folder, unless you intentionally want to publish the original source images too.

Before publishing, make sure every image is safe to redistribute and choose `--private` if the dataset should only be available to collaborators.

```bash
hf auth login
hf repo create ezhoureal/flux-aura-style --type dataset --private
hf upload ezhoureal/flux-aura-style training_data/flux_aura_style .
```

Other users can fetch it with:

```bash
hf download ezhoureal/flux-aura-style \
  --repo-type dataset \
  --local-dir training_data/flux_aura_style
```

Then they can train without running `lora prepare-dataset`:

```bash
uv sync
uv run lora train
```

The prepared dataset must keep this layout:

```text
training_data/flux_aura_style/
  train/
    metadata.jsonl
    pair-001.png
    label-001.png
  conditioning/
    pair-001.png
```

## 4. Train the LoRA

Start training with:

```bash
lora train
```

The launcher will:

- run the tracked `scripts/train_dreambooth_lora_flux_lowmem.py` trainer
- precompute all caption embeddings with CLIP/T5 before loading the FLUX transformer onto the GPU
- free CLIP/T5 before transformer LoRA training starts

The default config is intentionally conservative for a small dataset:

- rank `8`
- resolution `512`
- `adamw` with learning rate `1e-4`
- gradient checkpointing on
- latent caching on
- in-training validation off, so the trainer does not reload the full inference pipeline while training

## 5. Test inference

Once training finishes:

```bash
lora infer
```

Or pass a custom prompt:

```bash
lora infer "a side-profile portrait of a man in the style of zrlprfl, pastel haze, cinematic silhouette"
```

The sample image is written to:

```text
samples/flux-lora-test.png
```

## 6. Notes on Flux training

- `FLUX.1-dev` is gated on Hugging Face, so you must accept the model terms before downloads work.
- The official DreamBooth Flux trainer is text-to-image. It trains from the target image and `prompt` column; paired source images are preserved in `conditioning/` and referenced by metadata for future image-conditioned workflows, but this trainer does not consume them.
- Flux LoRA training is memory-heavy. This workspace defaults to 512px, rank 8, gradient checkpointing, latent caching, and cached prompt embeddings to fit a 32 GB GPU more reliably.
- If your GPU still runs out of memory, reduce `resolution`, reduce `rank`, or increase `gradient_accumulation_steps`.
