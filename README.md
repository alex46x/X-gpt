# Project Genesis

Project Genesis is a research codebase for building a decoder-only large
language model from randomly initialized weights with Python and PyTorch.
Development proceeds in reviewed phases; the current repository contains only
the Phase 1 project foundation.

No tokenizer, model, training, evaluation, or inference implementation exists
yet.

## Requirements

- Python 3.11 through 3.13
- [uv](https://docs.astral.sh/uv/)

## Setup

```console
uv sync --locked --group dev
```

## Verify

```console
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest
uv build
```

See [architecture](docs/architecture.md), [development standards](docs/development.md),
and the [roadmap](docs/roadmap.md) before contributing.

## License

Project Genesis is licensed under the MIT License.
