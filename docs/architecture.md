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

Phase 8 implements token-weighted validation metrics, perplexity, named
next-token task and coding cases, synchronized throughput measurements,
fingerprinted canonical reports, and regression gates. Generated-answer and
code-execution benchmarks wait for inference and a secure execution boundary.

Phase 9 implements validated per-layer KV caches, bounded autoregressive
generation, greedy and sampled decoding controls, immutable conversation state,
role-delimited prompt formatting, tokenizer/model reply composition, and
fingerprinted exact-match completion benchmarks. Untrusted code execution
remains outside the library until a hardened sandbox exists.

Phase 10 adds inference-only bundles and HTTP transport beside the inference
boundary. Container and operational files remain in the external `deployment`
directory. Bundles verify model configuration, weights, tokenizer identity,
package version, and checksums before serving. The stateless FastAPI service
provides bounded authenticated generation/chat, liveness/readiness, structured
request logs, and a non-root CPU container.

Phase 11 requires bundle provenance and enforces semantic runtime compatibility.
Fatal inference failures make a replica unready while preserving liveness for
diagnostics. A dependency-free load probe supplies explicit error and latency
gates. Tag-driven release automation verifies locked source, builds checksummed
artifacts, and records GitHub build provenance. Compatibility, security,
incident, recovery, and rollback contracts define the application boundary;
platform-specific orchestration remains outside the repository.

Phase 12 adds `project_genesis.experiment` as the top-level composition root.
It follows the existing dependency direction while connecting all implemented
subsystems into one atomic local run. Strict train/validation separation,
vocabulary and context compatibility checks, deterministic batch replay, final
evaluation, checkpointing, and provenance-bearing bundle creation are enforced
before the run directory is published.

Phase 13 adds `project_genesis.preflight` beside experiment orchestration. It
validates source integrity and cross-configuration compatibility, constructs the
configured decoder on PyTorch's storage-free meta device, and reports parameter,
schedule, persistent-state, and requested-device facts before an expensive run.
It intentionally does not predict activation memory or throughput without real
hardware measurements.
