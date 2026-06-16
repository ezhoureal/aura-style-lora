## Development Guidelines
- manage script config with hydra in separate config folder. Avoid using CLI arguments.
- when tweaking script args, you are allowed to append the arg values to the command directly, but make sure to update yaml config files afterwards, to keep it reproducible and tractable.
- Verify critical paths with unit tests and e2e tests, and add regression tests when fixing a bug.
- Use strict typing via Pyright. Ensure all typing errors are cleared after an edit.
- Use ruff to format all src files.
- avoid dynamic import like `require_import` as much as possible. aim for simplicity.
- Avoid wrapping imports in `try`/`except ImportError` just to customize dependency errors; use direct imports at module or function scope.
- avoid absolute paths that are not reproducible in other environments. Design the repo to be robust and reproduce-friendly. Most important examples: do not commit paths under `/home/...`, local dataset/cache paths, local checkpoint paths, local Python binaries, or sibling-repo script paths like `../stable-worldmodel/scripts/...` unless they are explicitly documented, configurable, and not required by defaults. Prefer repo-relative paths, Hydra config interpolation, package/module entrypoints, env vars with documented defaults, and manifests that record exact checkpoint/dataset revisions.
- Avoid using default arguments (x: int = 5) and Optional arguments (`| None` included) unless they are absolutely necessary, especially when defining function parameters.
- Avoid capability-probing control flow like nested `hasattr(...)` / fallback `if` branches in core logic. Prefer explicit typed protocols, adapters, or single-purpose helper methods so runtime behavior is clear and unsupported objects fail loudly.

## Caveats
- Sometimes codex is initiated with `proxychains`, which might mess with `uv` package downloads. You can disable it with `env -u LD_PRELOAD -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy`.
