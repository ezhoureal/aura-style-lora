# LoRA Training Workspace

This repo is a minimal local workspace for training a style transfer LORA on various diffusion models.

The training pipeline is "image to image". We always use a source photorealistic image and train the model to transfer its style.

The dataset is uploaded to `ezhoureal/aura_style` on huggingface. Additional data should also be appended to it.

We compare the performance of the following models:
