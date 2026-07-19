# Phase 12 - End-to-End Experiment Execution

## Phase overview

Phase 12 connects the existing dataset, preprocessing, tokenizer, model,
training, evaluation, checkpoint, and inference-bundle components through one
production command. It introduces orchestration only; subsystem behavior remains
owned and tested by its original package.

## Architecture

```text
six strict YAML configurations + local dataset sources
  -> verified source manifest
  -> cleaned train and validation records
  -> tokenizer trained on train records
  -> tokenized train and validation records
  -> randomly initialized GPT decoder
  -> configured optimization steps
  -> validation metrics
  -> checkpoint + deployable bundle + run manifest
  -> atomic run-directory publication
```

`project_genesis.experiment` is an application-level composition root. It may
import every subsystem it coordinates. Those lower-level packages never import
the experiment module.

## Interfaces

The Python interface is `run_experiment(...) -> ExperimentResult`. The console
entry point is:

```console
genesis-train \
  --dataset-config configs/dataset/experiment.yaml \
  --output artifacts/runs/run-001 \
  --source-revision COMMIT_SHA \
  --training-run-id run-001 \
  --device cuda
```

The remaining five subsystem configuration paths default to their files beneath
`configs/` and may be replaced explicitly.

## Artifact layout

```text
run-001/
|-- input-manifest.json
|-- processed-manifest.json
|-- tokenizer.json
|-- checkpoint.pt
|-- run.json
|-- configs/                 Exact configuration snapshots
`-- bundle/
    |-- manifest.json
    |-- model.yaml
    |-- model.pt
    `-- tokenizer.json
```

The destination must not exist. Work happens in a sibling temporary directory
and the complete directory is renamed into place only after evaluation,
checkpointing, and bundle creation succeed.

## Design decisions

- Training and validation sources are both required. Evaluation never silently
  reuses training data.
- Tokenizer training consumes only the training split.
- Model vocabulary must equal the vocabulary actually produced by tokenizer
  training.
- Training sequence length must fit the model context.
- Training batches replay deterministically across epochs until `max_steps`.
  Shuffling is deferred until the training configuration defines its exact
  reproducibility contract.
- The run manifest records source, run, processed dataset, tokenizer, bundle,
  configuration checksums, step, and evaluation identities.
- No workflow engine, database, tracking service, or distributed launcher is
  added. Native filesystem artifacts are sufficient for the current execution
  boundary.

## Failure behavior

Missing splits, undersized token streams, incompatible vocabulary sizes,
invalid configuration, training failures, or evaluation failures abort the run.
The temporary directory is removed and an existing destination is never
overwritten.

## Testing

The integration test creates real train and validation files and strict
configuration files, then crosses every phase boundary. It verifies the final
training step, checkpoint and tokenizer artifacts, run manifest identity,
bundle provenance, successful bundle loading, and immutable destination policy.

## Acceptance commands

```powershell
uv sync --locked --group dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest -W error
uv build
```

Future work begins with a representative corpus and measured training behavior.
Distributed input sharding, randomized sampling, experiment tracking, and remote
artifact storage should be added only when that run demonstrates the need.
