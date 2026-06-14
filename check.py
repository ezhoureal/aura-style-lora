from safetensors.torch import load_file
from pathlib import Path

for ckpt in [
    "checkpoint-000250",
    "checkpoint-000500",
    "checkpoint-001000",
    "checkpoint-001500",
    "checkpoint-002000",
]:
    p = (
        Path("outputs/ablation/local_edit_lora/sd35_medium")
        / ckpt
        / "sd3_input_projection_32ch.safetensors"
    )
    if not p.exists():
        continue

    state = load_file(p)
    w = state["weight"].float()

    print(
        ckpt,
        "shape", tuple(w.shape),
        "first16 mean/max",
        float(w[:, :16].abs().mean()),
        float(w[:, :16].abs().max()),
        "second16 mean/max",
        float(w[:, 16:].abs().mean()),
        float(w[:, 16:].abs().max()),
    )