## Coding Guidelines
Write clean, concise, idiomatic Python.

Style rules:

- Prefer simple functions over classes unless state or polymorphism is clearly needed.
- Do not add abstractions “for future extensibility” unless requested.
- Avoid unnecessary helper functions, wrapper classes, config objects, factories, registries, and custom exceptions.
- Keep the happy path obvious.
- Use standard library tools before adding dependencies.
- Use type hints for function signatures, but avoid over-engineered typing.
- Prefer list/dict comprehensions only when they stay readable.
- Avoid clever one-liners if they obscure intent.
- Do not catch broad exceptions unless there is a concrete recovery action.
- Do not add logging, retries, CLI parsing, environment handling, or validation unless requested.
- Keep comments sparse. Comment why, not what.
- Remove dead code, unused imports, unused variables, and redundant branches.
- Prefer returning values directly over storing temporary variables used once.
- Make the smallest correct change. Prefer deleting or simplifying existing code over adding new layers.


## Additional Tips
- manage script config with hierarchical Hydra in separate config folder. Avoid using CLI arguments.
- Use Pyright for type check.
- Use ruff to format all src files.
- Prefer direct type constructors like `str(...)`, `int(...)`, `float(...)`, and `bool(...)` over bloated type checks or one-off helper functions. Keep simple conversions simple.
