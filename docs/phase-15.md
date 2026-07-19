# Phase 15 — Deterministic Data Sampling

## 1. Phase Overview

Phase 15 gives training a deterministic shuffled record order for every epoch
and makes that order exactly replayable after checkpoint resume. It does not add
a sampler class, change the checkpoint format, or implement distributed
sampling.

## 2. Architecture Explanation

Training data now flows through two deliberately small layers:

```text
tokenized DatasetRecord values
  → seeded per-epoch record permutation
  → deterministic token packing
  → endless training microbatch stream
  → skip checkpointed microbatches_seen on resume
  → Trainer
```

`iter_token_batches` remains the finite, order-preserving packing primitive.
`iter_shuffled_token_batches` owns epoch permutation and repetition. The
experiment composition root supplies the configured seed and replays the stream
to the checkpointed position.

## 3. Folder Structure

No new package is required:

```text
src/project_genesis/
├── experiment.py
└── training/
    ├── __init__.py
    └── data.py
tests/
└── training/
    └── test_training.py
docs/
└── phase-15.md
```

## 4. File Structure

- `training/data.py` contains finite packing and the seeded epoch iterator.
- `training/__init__.py` exposes the new training-data interface.
- `experiment.py` composes the iterator with checkpointed progress.
- `test_training.py` verifies reproducibility, replay, variation, and failures.
- Architecture, roadmap, and user-facing documentation record the contract.

## 5. Dependency Graph

```text
datasets.DatasetRecord
  → training.data
  → project_genesis.experiment
```

The implementation uses Python's standard-library `random.Random`; no dependency
or configuration schema is added. Validation batching remains stable and
unshuffled.

## 6. Design Decisions

- A fresh permutation is derived from the training seed and zero-based epoch.
  The global random generator is never consumed by data ordering.
- Record shuffling happens before existing token packing. Token order within a
  record is preserved.
- The existing checkpointed `Trainer.microbatches_seen` value is the canonical
  stream position. A resumed run reconstructs the same stream and skips exactly
  that many microbatches.
- The checkpoint format remains unchanged, so Phase 14 checkpoints stay
  loadable.
- Validation is not shuffled because stable evaluation order makes bounded
  validation reproducible.
- Empty or token-insufficient training inputs fail before training can spin.
- Distributed rank partitioning is not guessed without a real multi-GPU
  topology and launcher.

## 7. Implementation

`iter_shuffled_token_batches` materializes the already in-memory record sequence
once. For each epoch it copies that sequence, shuffles the copy with a local RNG
seeded by `"<training-seed>:<epoch>"`, and delegates packing to
`iter_token_batches`.

The experiment runner passes `TrainingConfig.seed`. On resume, `islice` skips the
restored `microbatches_seen` count. This preserves the next batch without adding
duplicated sampler state that could disagree with the trainer checkpoint.

Replay cost grows with the number of consumed microbatches. Persisting a
separate epoch and offset is deliberately deferred until measured resume startup
time proves that extra state and migration logic are needed.

## 8. Unit Tests

Focused tests verify:

- identical seeds and offsets produce identical input and target tensors;
- another seed changes the sampled stream;
- replay from a saved microbatch offset is exact;
- negative seeds fail;
- an input without one complete sequence fails.

The existing interrupted experiment test additionally verifies that
`microbatches_seen` survives a real checkpoint and the run resumes to
completion.

## 9. Documentation

Phase 7 now distinguishes finite stable packing from epoch-level shuffled
sampling. Phase 12 records seeded training order. Phase 14 identifies the
microbatch counter as the complete single-process sampling state. The repository
architecture and roadmap mark Phase 15 complete.

## 10. Review Checklist

- [x] Training records are shuffled once per epoch.
- [x] The same seed produces the same batch stream.
- [x] Resume reconstructs the next microbatch exactly.
- [x] Validation order is unchanged.
- [x] No sampler class or checkpoint migration was introduced.
- [x] No runtime dependency or configuration field was introduced.
- [x] Invalid and undersized inputs fail clearly.
- [x] Focused and full repository verification pass.

## 11. Phase Summary

Single-process training now has deterministic seeded epoch sampling and exact
checkpoint replay. This closes the known Phase 14 data-order gap while
preserving the existing public checkpoint contract.

## 12. Next Phase Preview

The next implementation phase should be selected from actual training inputs and
hardware facts. Dataset sharding/token caches depend on the chosen corpus and
scale; CUDA/distributed work depends on the friend's NVIDIA GPU and topology.
Neither is added speculatively.
