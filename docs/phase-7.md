# Phase 7 — Training engine

## Phase overview

Phase 7 turns tokenized dataset records into deterministic next-token updates for
the Phase 6 decoder. It owns training configuration, batching, language-model
loss, AdamW, learning-rate scheduling, gradient accumulation and clipping,
mixed precision, seeding, and atomic resumable checkpoints.

## Architecture

```text
tokenized DatasetRecord values
  → deterministic packing with an explicit separator token
  → shifted input and target batches
  → GPTDecoder logits
  → cross-entropy loss
  → accumulated and clipped gradients
  → AdamW update
  → warmup/cosine learning-rate update
```

Training depends on dataset and model contracts. Neither subsystem imports the
training package.

## Configuration

`TrainingConfig` validates batch dimensions, AdamW hyperparameters, warmup and
cosine-decay bounds, accumulation, clipping, precision, and the experiment seed.
The strict default YAML is `configs/training/default.yaml`.

Phase 14 extends that strict policy with positive checkpoint, evaluation, and
logging intervals plus checkpoint retention.

The configured sequence length must fit the selected model context window. That
compatibility remains visible at model execution rather than being hidden in a
combined experiment schema.

## Data batching

`iter_token_batches` streams encoded records in the supplied order. A separator
token ID is mandatory because the tokenizer default does not append an end
token. Packing retains the transition between adjacent windows and can either
drop an incomplete final batch or return its complete rows.

Phase 15 adds `iter_shuffled_token_batches`, which derives a fresh deterministic
record permutation from the configured seed and epoch before delegating to the
finite packer. A caller resumes the same stream by skipping
`Trainer.microbatches_seen` batches with `itertools.islice`. Distributed rank
partitioning remains deferred until a multi-GPU launcher is exercised.

## Optimization and scheduling

AdamW applies weight decay only to parameters with at least two dimensions;
biases and normalization parameters are excluded. The scheduler uses native
PyTorch linear warmup followed by cosine decay to the configured minimum ratio.

`Trainer.train_step` requires exactly the configured number of microbatches,
divides each loss before backpropagation, rejects non-finite loss and gradients,
clips the accumulated gradient norm, and advances the schedule only after a
successful optimizer update.

## Precision and devices

Float32 works on CPU and CUDA. Bfloat16 uses native autocast on supported devices.
Float16 is restricted to CUDA and uses `torch.amp.GradScaler`; an overflow-skipped
update does not advance the optimizer-step counter or scheduler. Source remains
device-agnostic and CI remains CPU-only.

The trainer accepts a regular module or a module already wrapped by native
`DistributedDataParallel`. Project-specific process launching and distributed
sampling are deferred until multi-GPU infrastructure and CUDA CI exist; native
`torchrun` and PyTorch DDP cover orchestration without a repository-specific
framework.

## Checkpoints and deterministic resumption

Checkpoints contain a format version, model, optimizer, scheduler, scaler,
optimizer-step and microbatch counters, and PyTorch CPU/CUDA RNG states. Loading
uses PyTorch's restricted weights-only mode and rejects unexpected structure.

Saving writes and synchronizes a temporary file in the destination directory,
then atomically replaces the target. A failed write removes its temporary file.
Model and optimizer compatibility is enforced by native state loading.

## Testing

Tests cover strict configuration, explicit document separators, rejection of raw
records, accumulation equivalence, CPU precision policy, atomic checkpoint
loading, and bit-exact continuation of the next dropout-enabled training step.

## Acceptance commands

```powershell
uv sync --locked --group dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest -W error
uv build
```
