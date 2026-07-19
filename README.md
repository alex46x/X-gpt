# Project Genesis

Project Genesis is a research codebase for building a decoder-only large
language model from randomly initialized weights with Python and PyTorch.
Development proceeds in reviewed phases.

Phases 2 and 3 provide:

- Safe typed YAML configuration with strict dotted overrides.
- Runtime environment detection and configuration-relative paths.
- Immutable dataset records, schemas, metadata, and statistics.
- Deterministic local manifests, SHA-256 fingerprints, and integrity checks.
- Dataset registry, cache contract, and atomic local manifest storage.
- Deterministic readers for text, Markdown, JSON, JSONL, CSV, PDF, Git snapshots,
  and local HTML snapshots.
- Configurable normalization, filtering, exact deduplication, quality reports,
  and processed-data manifests.

Tokenizer, model, training, evaluation, and inference behavior remain out of
scope.

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)

## Setup

```console
uv sync --locked --group dev
```

The default dataset configuration is
[`configs/dataset/default.yaml`](configs/dataset/default.yaml). It deliberately
contains no sources; experiments must declare their local inputs explicitly.
Cleaning defaults are in
[`configs/preprocessing/default.yaml`](configs/preprocessing/default.yaml).

## Verify

```console
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest
uv build
```

See the [architecture](docs/architecture.md), [Phase 2 decisions](docs/phase-2.md),
[Phase 3 decisions](docs/phase-3.md), [development standards](docs/development.md),
and [roadmap](docs/roadmap.md).

## License

Project Genesis is licensed under the MIT License.
