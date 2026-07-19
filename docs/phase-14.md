# Phase 14 - Recoverable Training Runs

## Phase overview

Phase 14 makes the single-process experiment runner safe for long training runs.
It adds periodic checkpoints, latest-checkpoint resume, periodic validation,
durable JSONL metrics, best-checkpoint selection, checkpoint retention, and
graceful interruption without adding a tracking service or scheduler.

## Architecture

```text
configured experiment
  -> .<run>.in-progress/
       |-- state.json
       |-- metrics.jsonl
       |-- best-checkpoint.json
       |-- checkpoints/step-N.pt
       `-- immutable preparation artifacts
  -> train / validate / checkpoint
  -> interruption: preserve staging and latest checkpoint
  -> --resume latest checkpoint
  -> final evaluation, checkpoint, and inference bundle
  -> atomic rename to final run directory
```

The final destination remains immutable. An unfinished run has one deterministic
sibling staging directory, so its checkpoint and identity can be found without a
database.

## Training configuration

`TrainingConfig` now requires:

- `checkpoint_interval_steps`
- `evaluation_interval_steps`
- `log_interval_steps`
- `keep_last_checkpoints`

Every value is a positive integer. Final evaluation and checkpointing happen
regardless of interval alignment.

## Resume contract

Resume uses:

```console
genesis-train [same arguments] \
  --output artifacts/runs/run-001 \
  --resume artifacts/runs/.run-001.in-progress/checkpoints/step-00001000.pt
```

Only the latest checkpoint in the matching staging directory is accepted. The
source revision, training-run ID, dataset fingerprint, tokenizer fingerprint,
and all six configuration checksums must match. The runner restores model,
optimizer, scheduler, scaler, CPU/CUDA RNG, optimizer step, and microbatch
position through the existing checkpoint contract.

The deterministic batch stream skips `microbatches_seen` entries. Seeded
per-epoch shuffling is implemented in Phase 15; the microbatch counter is the
complete single-process sampling position, so no separate sampler state is
required.

## Metrics and best checkpoint

`metrics.jsonl` contains compact records:

```json
{"learning_rate":0.001,"loss":5.1,"step":10,"type":"train"}
{"loss":4.9,"perplexity":134.3,"step":100,"type":"validation"}
```

Each record is flushed and synchronized. Resume truncates records newer than the
checkpoint so it cannot preserve metrics for rolled-back work.

Validation checkpoints compete on lowest loss. `best-checkpoint.json` points to
the selected periodic checkpoint instead of duplicating model weights.

Retention preserves the latest configured number of checkpoints plus the best
checkpoint when it is older.

## Failure behavior

`Ctrl+C` and SIGTERM save an emergency checkpoint, mark the staging run
interrupted, print the exact resume path, and re-raise the interruption. Other
runtime failures mark the run failed and preserve its last periodic checkpoint
for inspection or explicit resume.

Preparation failures remove a newly created staging directory. Existing resume
artifacts are never deleted after an identity or compatibility failure.

## Testing

The complete experiment integration test interrupts immediately after the first
optimizer step, verifies the emergency state and checkpoint, resumes from that
checkpoint, completes the configured step count, enforces retention, publishes
unique metrics, and loads the final inference bundle.

## Acceptance commands

```powershell
uv sync --locked --group dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest -W error
uv build
```

Skipped intentionally: arbitrary old-checkpoint rollback, remote artifact
storage, background checkpoint workers, W&B/MLflow, and distributed rank
coordination. Add them only when the selected dataset or GPU topology requires
them.
