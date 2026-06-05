---
license: other
base_model:
- black-forest-labs/FLUX.2-dev
library_name: diffusers
pipeline_tag: image-to-image
tags:
- flux
- flux-2
- flux-2-edit
- lora
- image-editing
- style-transfer
- aura-style
---

# Aura Style LoRA for Flux-2 Edit

This repository contains a LoRA adapter trained for Flux-2 Edit image editing. It applies a radiant aura style with smooth colorful gradients, ethereal haze, subtle contour lighting, and a refined cinematic glow while preserving the input subject and composition.

The main adapter file is:

```text
pytorch_lora_weights.safetensors
```

These are FAL-format Flux-2 Edit LoRA weights produced by `fal-ai/flux-2/lora/edit`.

A preconverted Diffusers-format copy is also provided:

```text
pytorch_lora_weights.diffusers.safetensors
```

## Training Summary

- Base/edit model: Flux-2 Edit
- Training service: FAL Flux-2 LoRA Edit trainer
- Steps: 1,000
- Learning rate: 5e-5
- Adapter format: FAL Flux-2 Edit LoRA safetensors
- Dataset: 25 paired edit examples plus metadata
- Style target: radiant colorful aura lighting, smooth gradients, haze, contour highlights, cinematic glow

Default training instruction:

```text
Apply a radiant aura lighting style with smooth colorful gradients, ethereal haze, subtle contour lighting, and a refined cinematic glow while preserving the subject and composition.
```

## Repository Layout

```text
pytorch_lora_weights.safetensors
pytorch_lora_weights.diffusers.safetensors
config_342740f8-8ed7-4b9a-88c3-7afb87679d59.json
fal-training-result.json
training_data/
  train/
    metadata.jsonl
    pair-001.png
    ...
  conditioning/
    pair-001.png
    ...
```

`training_data/train/` contains the target styled images and metadata. `training_data/conditioning/` contains the corresponding source images used for image-edit training.

## FAL Usage

Use this repository as the LoRA source with the Flux-2 LoRA Edit endpoint. A typical edit prompt is:

```text
Transform this photorealistic image into the trained radiant aura style: smooth colorful gradients, ethereal haze, subtle contour lighting, and a refined cinematic glow. Preserve the subject identity, composition, pose, silhouette, camera framing, and important details.
```

Suggested inference settings:

- LoRA scale: `1.0`
- Guidance scale: `2.5`
- Inference steps: `28`

## Local Diffusers Usage

The uploaded `pytorch_lora_weights.safetensors` file is in FAL format. For local Diffusers inference, convert it to Diffusers key format before loading it.

This repository was validated with a helper script that converts the FAL keys and checks the converted adapter against `Flux2Transformer2DModel`:

```bash
python scripts/run_flux2_edit_lora_local.py --check-only \
  --lora fal_flux2_edit_lora/pytorch_lora_weights.safetensors
```

After conversion, load the converted adapter with Diffusers:

```python
from diffusers import Flux2Pipeline

pipe = Flux2Pipeline.from_pretrained("diffusers/FLUX.2-dev-bnb-4bit")
pipe.load_lora_weights("pytorch_lora_weights.diffusers.safetensors", adapter_name="aura")
pipe.set_adapters(["aura"], adapter_weights=[1.0])
```

## Intended Use

This adapter is intended for stylized image editing where the source subject and composition should remain recognizable while the output receives a luminous aura treatment. It works best with clear source images, simple-to-medium complexity compositions, and prompts that explicitly ask to preserve identity, pose, framing, and important details.

## Limitations

- This LoRA is style-focused and may over-apply glow or color gradients at high LoRA scales.
- It was trained on a compact paired dataset, so unusual domains may need prompt tuning or a lower adapter strength.
- The adapter is appended to Flux-2 Edit behavior; base model terms and access requirements still apply.
