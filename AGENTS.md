## Constraints
- minimize VRAM and RAM usage while preserving high-quality LORA training. <32GB VRAM is the hard constraint, ideally <24GB

## Development Guidelines
- After edits finish, always ensure `ruff check` and `pyright` type check passes on all src files in the repo.
- Run smoke tests to verify the training/inference behavior