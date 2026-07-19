# Architecture

## Goal

Project Genesis will implement a decoder-only large language model from randomly
initialized weights. PyTorch is the tensor and automatic-differentiation runtime;
the tokenizer, model architecture, training loop, evaluation, and inference
behavior are owned by this repository.

The repository uses a `src` layout. A subsystem is added only when its phase
introduces working behavior and tests.

## Target repository

```text
project-genesis/
├── .github/workflows/       Continuous integration
├── configs/                 Versioned YAML experiment configurations
├── deployment/              Containers and deployment manifests
├── docs/                    Architecture and engineering documentation
├── scripts/                 Thin, task-oriented command entry points
├── src/project_genesis/
│   ├── configuration/       Typed configuration loading and validation
│   ├── datasets/            Dataset records, readers, and composition
│   ├── preprocessing/       Cleaning, normalization, and filtering
│   ├── tokenizer/           Vocabulary and tokenizer training/encoding
│   ├── model/               Embeddings, attention, blocks, and decoder
│   ├── training/            Trainer, optimizer, scheduler, and checkpoints
│   ├── evaluation/          Metrics and evaluation orchestration
│   ├── inference/           Autoregressive generation
│   ├── chat/                Conversation state and prompt assembly
│   ├── benchmark/           Reproducible quality and performance benchmarks
│   └── utilities/           Small shared primitives with multiple consumers
└── tests/                   Tests mirroring implemented source subsystems
```

Future names describe boundaries, not pre-approved abstractions. Files and
packages are created only when their implementation phase needs them.

## Responsibilities

- **Configuration:** typed schemas, safe YAML loading, validation, and path
  resolution.
- **Datasets:** immutable records, source readers, streaming, sharding, and
  dataset composition.
- **Preprocessing:** deterministic cleaning, normalization, deduplication, and
  quality filtering.
- **Tokenizer:** vocabulary construction, tokenizer training, encoding, decoding,
  and special-token policy.
- **Model:** token and position embeddings, causal self-attention, feed-forward
  layers, normalization, residual connections, transformer blocks, and the
  decoder-only network.
- **Training:** batching, loss computation, optimization, scheduling,
  distributed execution, mixed precision, and checkpoint lifecycle.
- **Evaluation and benchmark:** loss and task metrics, regression gates, coding
  evaluations, and reproducible performance measurements.
- **Inference and chat:** autoregressive decoding, sampling, conversation state,
  and serving-facing contracts.
- **Deployment:** operational packaging around inference; it is not importable
  application logic.

Coding capability is an outcome of coding data, objectives, model behavior, and
benchmarks. It is not a separate model abstraction.

## Dependency direction

```text
configuration ─────────────────────────────────────────────┐
utilities ─────────────────────────────────────────────────┤
datasets → preprocessing → tokenizer                       │
tokenizer + configuration → model                          │
datasets + tokenizer + model → training                    │
model + tokenizer + checkpoint → inference → chat          │
datasets + model + inference → evaluation → benchmark      │
inference + configuration → deployment                     │
```

Imports must follow these rules:

1. Lower-level packages never import orchestration or deployment packages.
2. Deployment may import the public inference and configuration interfaces;
   library code never imports deployment.
3. Tests may cross boundaries only to exercise an explicit integration path.
4. Circular imports are architecture defects and must be removed rather than
   hidden behind local imports.
5. A shared helper belongs in `utilities` only after at least two subsystems use
   it.

## Configuration contract

Experiment configuration will use YAML. Each consuming subsystem owns a typed
schema and validates values at the load boundary. Loaders must use safe YAML
parsing. Relative paths are resolved from the configuration file that declares
them, not from the process working directory. Unknown fields fail validation so
misspellings cannot silently alter experiments.

PyYAML will be added when the configuration loader is implemented. No generic
configuration dictionary is introduced in advance of typed consumers.

## Compute policy

Training targets NVIDIA CUDA, while source code remains device-agnostic wherever
PyTorch supports it. Foundational CI is CPU-only and must not require accelerator
drivers. CUDA installation instructions and accelerator tests will be added with
the first model/training phase that uses them. ROCm and Apple MPS are not support
commitments until explicitly adopted and tested.

## Artifact policy

Datasets, checkpoints, experiment runs, and model weights are generated artifacts
and are excluded from Git. Versioned metadata and small deterministic fixtures may
be committed. Later phases must define checksums and atomic writes before storing
valuable artifacts.

## Implemented foundation

Phase 2 implements `configuration` and `datasets`. Configuration is independent;
datasets imports its loader and shared path and environment models. Dataset source
formats are declarations only: no tokenizer, content parser, cleaner, model, or
training behavior exists.

Manifest construction inventories local files deterministically and verifies
their size and SHA-256 content. Immutable records define provenance and reserve
optional fields for later derived values. The in-memory registry and local atomic
manifest store cover current consumers without committing the project to a
distributed registry or cache.

Phase 3 implements `preprocessing`, which depends on dataset contracts and emits
the same immutable `Dataset` and `DatasetRecord` types. Reader selection and raw
manifest inventory share one deterministic file-selection function so ignored
files cannot silently influence raw fingerprints.

Phase 4 implements a custom byte-level BPE `tokenizer`. It consumes cleaned
datasets, guarantees UTF-8 coverage through byte IDs, and emits new immutable
records with `token_ids` populated. Tokenizer code has no model or PyTorch
dependency.

Phase 5 implements independently tested PyTorch `model` primitives with
batch-first tensor contracts. Phase 6 composes them into pre-normalization
transformer blocks and a decoder-only token-to-logits model.

Phase 7 implements deterministic token packing, next-token loss, AdamW,
warmup/cosine scheduling, accumulated and clipped gradients, native mixed
precision, and atomic training checkpoints. Multi-process execution uses native
PyTorch DDP when deployed; project-specific launch orchestration waits for an
exercised multi-GPU environment.
