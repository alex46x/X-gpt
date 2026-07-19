# Development

## Environment

Project Genesis supports Python 3.11 through 3.13 and uses `uv` as the single
dependency and environment workflow.

```console
uv sync --locked --group dev
```

`pyproject.toml` declares direct dependencies and tool settings. `uv.lock`
records the complete reproducible resolution and must be committed whenever
dependencies change. Do not add `requirements.txt`, Conda manifests, or a second
package metadata file.

Add a runtime dependency only in the phase that imports it:

```console
uv add PACKAGE
```

Add a development-only dependency with:

```console
uv add --group dev PACKAGE
```

## Required checks

Run these commands before review:

```console
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest
uv build
```

To apply formatting locally, run `uv run ruff format .`. CI checks but never
rewrites files.

## Python standards

- Public modules, classes, and functions require useful docstrings.
- Public function signatures and dataclass fields require explicit types.
- Mypy strict mode is the default; narrow exceptions require an inline reason.
- Prefer immutable dataclasses for validated records and configuration values.
- Validate external data at its boundary and keep trusted internal paths simple.
- Use `pathlib.Path`; never embed machine-specific paths.
- Use the standard library before adding dependencies.
- Keep side effects in orchestration layers and numerical kernels independently
  testable.
- Do not add factories, protocols, registries, or base classes until multiple
  implementations require them.
- Never commit secrets, datasets, checkpoints, weights, or experiment output.

## Tests

Tests mirror implemented source behavior. New non-trivial logic requires the
smallest test that would fail if that behavior regressed. Numerical tests must
state tolerances and seed randomness. Tests must run on CPU unless they are
explicitly marked as accelerator integration tests in a future phase.

Unit tests must not download data or contact external services. Small,
deterministic fixtures belong under `tests`; large fixtures belong in managed
artifact storage introduced by a later phase.

## Change discipline

Each phase is reviewed before the next begins. A change must:

1. Stay within the active phase.
2. Preserve the documented dependency direction.
3. Include its implementation, tests, configuration, and documentation together.
4. Pass the same locked checks locally and in CI.

Git history, branch policy, release automation, and remote hosting policy are
left to the repository owner; Phase 1 creates no commit or remote.
