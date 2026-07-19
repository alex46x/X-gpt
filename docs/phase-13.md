# Phase 13 - Training Preflight and Capacity Report

## Phase overview

Phase 13 validates a proposed baseline training run before tokenizer training,
model allocation, or optimization begins. It reports source, model, schedule,
and requested-device facts without claiming that a baseline run occurred.

## Architecture

```text
six strict YAML configurations
  -> dataset source inventory and checksum verification
  -> train/validation file validation
  -> tokenizer/model vocabulary compatibility
  -> sequence/context compatibility
  -> meta-device model construction
  -> parameter and persistent-state lower bound
  -> requested-device availability
  -> strict JSON readiness report
```

`project_genesis.preflight` is a top-level composition module. It reuses existing
configuration, dataset-manifest, model, and training contracts without changing
their dependency direction.

## Interface

```console
genesis-preflight \
  --dataset-config configs/dataset/experiment.yaml \
  --device cuda
```

The Python interface is:

```python
preflight_experiment(...) -> PreflightReport
```

The command exits nonzero when the requested device is unavailable or the
persistent training-state lower bound alone exceeds reported device memory.
Invalid sources and incompatible configurations fail before a report is emitted.

## Report fields

- Dataset manifest fingerprint and source file/byte counts.
- Training and validation file counts.
- Configured vocabulary size.
- Exact model parameter count.
- Persistent training-state lower bound.
- Tokens per optimizer step and total scheduled tokens.
- Requested device, availability, reported total memory, and readiness.

The persistent-state lower bound is 16 bytes per parameter: four bytes each for
model weights and gradients plus eight bytes for Adam's two moment tensors. It
does not include activations, attention intermediates, allocator fragmentation,
temporary optimizer buffers, framework state, or other processes. Passing this
check does not prove that the complete workload fits.

## Design decisions

- PyTorch's native `meta` device preserves tensor shapes and exact parameter
  counting without allocating model storage.
- Preflight inventories and hashes actual declared files but does not parse,
  clean, or tokenize the corpus. The real experiment remains the authoritative
  content validation.
- CPU availability is reported, but total CPU memory is left unknown instead of
  adding platform-specific memory probing.
- CUDA capacity comes from PyTorch only when the requested device exists.
- Cloud recommendations, synthetic throughput, and guessed activation formulas
  are excluded because they would create false precision.

## Local environment result

The current workstation exposes only the Microsoft basic display adapter, and
the committed PyTorch environment is CPU-only. A CUDA baseline cannot be
executed or honestly characterized here.

## Testing

The existing complete experiment fixture now runs preflight first. It verifies
source split counts, parameter and persistent-state calculations, scheduled
tokens, CPU readiness, and early rejection of tokenizer/model vocabulary
mismatch.

## Acceptance commands

```powershell
uv sync --locked --group dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest -W error
uv build
```

The next executable milestone requires a real licensed dataset configuration and
a CUDA-enabled dependency lock on known hardware.
