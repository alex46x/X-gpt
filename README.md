# Project Genesis

Project Genesis is a research codebase for building a decoder-only large
language model from randomly initialized weights with Python and PyTorch.
Development proceeds in reviewed phases.

Phases 2 through 13 provide:

- Safe typed YAML configuration with strict dotted overrides.
- Runtime environment detection and configuration-relative paths.
- Immutable dataset records, schemas, metadata, and statistics.
- Deterministic local manifests, SHA-256 fingerprints, and integrity checks.
- Dataset registry, cache contract, and atomic local manifest storage.
- Deterministic readers for text, Markdown, JSON, JSONL, CSV, PDF, Git snapshots,
  and local HTML snapshots.
- Configurable normalization, filtering, exact deduplication, quality reports,
  and processed-data manifests.
- Custom byte-level BPE vocabulary training, reversible Unicode encoding,
  stable special-token IDs, tokenizer persistence, and quality metrics.
- PyTorch token and position embeddings, custom LayerNorm, feed-forward layers,
  causal multi-head self-attention, and strict residual addition.
- Pre-normalization transformer blocks and a GPT-style decoder.
- Deterministic next-token batching, AdamW, warmup/cosine scheduling, gradient
  accumulation and clipping, mixed precision, and atomic resumable checkpoints.
- Token-weighted validation, perplexity, named language and coding cases,
  throughput measurements, canonical reports, and regression gates.
- Cached autoregressive generation, greedy and stochastic sampling, stop/context
  handling, immutable conversations, prompt assembly, and completion benchmarks.
- Verified inference-only bundles, bounded generate/chat HTTP APIs,
  health/readiness checks, structured request logs, and a non-root CPU container.
- Required bundle provenance and semantic compatibility checks.
- Fatal inference failure isolation, a dependency-free concurrent load probe,
  operational recovery procedures, and checksummed attested releases.
- Atomic end-to-end experiment execution from verified local sources through a
  trained, evaluated, checkpointed, deployable inference bundle.
- A non-allocating training preflight for source integrity, configuration
  compatibility, parameter count, schedule size, and device capacity facts.

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
Tokenizer defaults are in
[`configs/tokenizer/default.yaml`](configs/tokenizer/default.yaml).
Model defaults are in [`configs/model/default.yaml`](configs/model/default.yaml).
Training defaults are in
[`configs/training/default.yaml`](configs/training/default.yaml).
Evaluation defaults are in
[`configs/evaluation/default.yaml`](configs/evaluation/default.yaml).
Inference defaults are in
[`configs/inference/default.yaml`](configs/inference/default.yaml).

Run a complete experiment with a dataset configuration that declares both
training and validation sources:

```console
genesis-preflight \
  --dataset-config configs/dataset/experiment.yaml \
  --device cuda
```

```console
genesis-train \
  --dataset-config configs/dataset/experiment.yaml \
  --output artifacts/runs/run-001 \
  --source-revision COMMIT_SHA \
  --training-run-id run-001 \
  --device cuda
```

## Verify

```console
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest
uv build
```

See the [architecture](docs/architecture.md), [Phase 2 decisions](docs/phase-2.md),
[Phase 3 decisions](docs/phase-3.md), [Phase 4 decisions](docs/phase-4.md),
[Phase 5 decisions](docs/phase-5.md), [Phase 6 decisions](docs/phase-6.md),
[Phase 7 decisions](docs/phase-7.md), [Phase 8 decisions](docs/phase-8.md),
[Phase 9 decisions](docs/phase-9.md), [Phase 10 decisions](docs/phase-10.md),
[Phase 11 decisions](docs/phase-11.md), [compatibility policy](docs/compatibility.md),
[Phase 12 decisions](docs/phase-12.md),
[Phase 13 decisions](docs/phase-13.md),
[production runbook](docs/runbook.md), [security policy](SECURITY.md),
[development standards](docs/development.md), and the [roadmap](docs/roadmap.md).

## License

Project Genesis is licensed under the MIT License.
