# Phase 8 — Evaluation and benchmarks

## Phase overview

Phase 8 measures the current decoder's teacher-forced quality, named next-token
behavior, and execution throughput. It produces immutable, fingerprinted reports
and deterministic regression decisions without depending on generation or
external benchmark services.

## Architecture

```text
model + validation batches
  → token-weighted evaluation
  → loss, perplexity, accuracy, throughput

model + ordered benchmark cases
  → final-position next-token predictions
  → case results + suite fingerprint

evaluation + benchmark + metadata
  → canonical JSON report
  → regression gate
```

`evaluation` depends on the model interface and training's existing `TokenBatch`
contract. `benchmark` depends on evaluation results and the existing atomic text
writer. Model and training packages do not import either subsystem.

## Configuration

`EvaluationConfig` owns the maximum validation-batch count and
`RegressionThresholds`. The strict default YAML is
`configs/evaluation/default.yaml`.

Regression thresholds bound absolute loss increase, absolute token-accuracy
drop, and throughput relative to an accepted baseline. Throughput comparisons
are meaningful only on equivalent hardware, software, batch shapes, and runtime
conditions.

## Corpus evaluation

`evaluate_model` uses `torch.inference_mode` and temporarily switches the model
to evaluation mode, restoring its original mode even if evaluation fails.
Cross-entropy is summed per token before aggregation, avoiding incorrect averages
when batch sizes differ.

Reported metrics are:

- Mean next-token cross-entropy.
- Perplexity derived from mean loss.
- Exact next-token accuracy.
- Evaluated token and batch counts.
- Monotonic elapsed time and tokens per second.

CUDA devices are synchronized immediately around timed work so queued kernels do
not produce misleading measurements. Evaluation rejects malformed and empty
batches.

## Named task and coding benchmarks

`BenchmarkCase` stores a unique name, category, prompt token IDs, and expected
next token. Categories can distinguish language, coding, or experiment-specific
tasks without creating category subclasses.

The ordered suite receives a canonical SHA-256 fingerprint over every prompt and
expected token. Results retain category, prediction, correctness, timing, and
suite identity. One forward pass per case keeps variable-length behavior obvious;
length bucketing should be added only when measured suite runtime requires it.

Generated completion scoring and code execution are deferred because
autoregressive inference and a secure execution sandbox do not exist yet.

## Reports and regression gates

`BenchmarkReport` combines evaluation metrics, benchmark results, and immutable
string metadata. Canonical JSON uses sorted keys, compact separators, UTF-8, and
strict finite numbers. Reports have their own SHA-256 fingerprint and are saved
through the existing flushed atomic text writer.

Loading rejects malformed JSON, unknown or missing fields, unsupported versions,
invalid metric bounds, inconsistent predictions, and invalid suite fingerprints.

`compare_results` returns every loss, accuracy, and throughput regression rather
than stopping at the first failure, making it suitable for CI reporting.

## Testing

Tests use a deterministic next-token model to verify token-weighted aggregation,
accuracy, batch limits, timing, mode restoration, task and coding categories,
suite identity, regression pass/fail behavior, strict report loading, and
canonical atomic report round trips.

## Acceptance commands

```powershell
uv sync --locked --group dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest -W error
uv build
```
