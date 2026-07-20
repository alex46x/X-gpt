# Phase 16 — Reproducible Coding Smoke Corpus

## 1. Phase Overview

Phase 16 turns the three user-selected repositories into a pinned, licensed,
reproducible smoke experiment. It validates the complete dataset-to-inference
path without claiming that this small corpus produces a capable coding model.

## 2. Architecture Explanation

```text
pinned upstream Git commits
  → ignored local snapshots
  → existing Git source reader
  → train/validation preprocessing
  → 320-token custom BPE
  → 850,688-parameter decoder
  → 20-step recoverable smoke run
  → verified inference bundle
```

No upstream code is imported or executed.

## 3. Folder Structure

```text
configs/{dataset,tokenizer,model,training,evaluation}/coding-smoke.yaml
scripts/prepare_coding_smoke.py
data/coding-smoke/                 Generated and Git-ignored
artifacts/runs/coding-smoke-cpu/   Generated and Git-ignored
docs/phase-16.md
```

## 4. File Structure

- The preparation script fetches and verifies exact commits.
- The dataset config selects text/code extensions and explicit licenses.
- The four subsystem configs define a deliberately small runnable experiment.
- This document records provenance, commands, results, and limitations.

## 5. Dependency Graph

```text
Git executable → pinned snapshots → existing dataset/preprocessing APIs
                                → tokenizer → model/training → inference bundle
```

The source package and runtime dependency set are unchanged.

## 6. Design Decisions

- nanoGPT and minGPT are training sources; lit-llama is validation-only.
- Repository-level separation prevents validation files from entering training.
- Only `.json`, `.md`, `.py`, `.sh`, `.toml`, `.txt`, `.yaml`, and `.yml`
  files are selected. Git metadata, notebooks, images, and generated binaries
  are excluded.
- The corpus remains outside Git; one script restores it on another machine.
- The smoke model uses float32 so it runs on CPU before CUDA selection.
- No generic downloader, dataset catalog, or automatic licensing policy is
  introduced.

## 7. Implementation

Reviewed snapshots:

| Repository | Split | License | Commit |
| --- | --- | --- | --- |
| `karpathy/nanoGPT` | train | MIT | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` |
| `karpathy/minGPT` | train | MIT | `37baab71b9abea1b76ab957409a1cc2fbfba8a26` |
| `Lightning-AI/lit-llama` | validation | Apache-2.0 | `2a464de2a1d2f266614d15091d3d7f30330c3ede` |

Materialize and preflight:

```console
uv run --locked python scripts/prepare_coding_smoke.py
uv run --locked genesis-preflight \
  --dataset-config configs/dataset/coding-smoke.yaml \
  --tokenizer-config configs/tokenizer/coding-smoke.yaml \
  --model-config configs/model/coding-smoke.yaml \
  --training-config configs/training/coding-smoke.yaml \
  --evaluation-config configs/evaluation/coding-smoke.yaml \
  --device cpu
```

Train by passing the same five smoke configs to `genesis-train`, plus an output,
source revision, training-run ID, and device.

## 8. Unit Tests

The preparation script was run against all three existing clones and verified
their origin URLs and exact commits. Repository preflight then hashed the real
selected files and validated all subsystem contracts. The complete 20-step CPU
experiment exercised tokenizer training, model training, periodic evaluation,
checkpointing, bundle publication, bundle verification, and cached generation.

## 9. Documentation

The README contains copyable preparation, preflight, and training commands.
Generated corpus and run artifacts remain excluded by the existing `.gitignore`.

## 10. Review Checklist

- [x] Every upstream source has an explicit license.
- [x] Exact commits are recorded and reproducible.
- [x] Training and validation repositories are disjoint.
- [x] Third-party code is treated only as text.
- [x] Preflight reports ready.
- [x] A complete CPU smoke run publishes a verified bundle.
- [x] No new package dependency or source abstraction was added.

## 11. Phase Summary

The first real coding corpus is reproducible and the entire local experiment
path has run successfully.

## 12. Next Phase Preview

This corpus is only a systems smoke test. A useful coding model requires a much
larger, deliberately licensed and deduplicated corpus. CUDA configuration still
waits for the target NVIDIA GPU and driver facts.
