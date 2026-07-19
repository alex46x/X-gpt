# Roadmap

Every phase ends with working code, focused tests, documentation, and review
before the next phase begins. Phase boundaries may be refined when measured
requirements emerge, but they must not be skipped.

## Phase 1 — Repository foundation

Installable package, locked development environment, CI, engineering standards,
and target architecture. No language-model behavior.

## Phase 2 — Configuration and dataset foundations

Typed YAML configuration, dataset record contracts, source manifests, validation,
and deterministic local ingestion. No tokenizer or model behavior.

Status: complete.

## Phase 3 — Data cleaning and preprocessing

Streaming normalization, filtering, deduplication, quality reporting, and
reproducible processed-data manifests.

Status: complete.

## Phase 4 — Vocabulary and tokenizer

Vocabulary construction, special-token policy, tokenizer training, deterministic
encoding/decoding, serialization, and tokenizer quality tests.

Status: complete.

## Phase 5 — Model primitives

Token embeddings, positional encoding, normalization, feed-forward layers,
causal multi-head self-attention, and residual connections in PyTorch.

Status: complete.

## Phase 6 — Decoder-only architecture

Transformer block composition, GPT-style decoder, initialization, parameter
accounting, masking, forward contracts, and numerical tests.

Status: complete.

## Phase 7 — Training engine

Batching, language-model loss, optimizer and scheduler creation, mixed precision,
gradient accumulation and clipping, distributed execution, deterministic
resumption, and atomic checkpoints.

Status: complete.

## Phase 8 — Evaluation and benchmarks

Validation loss, perplexity, task and coding benchmarks, performance measurements,
regression thresholds, and reproducible reports.

Status: complete.

## Phase 9 — Inference and chat

Autoregressive decoding, sampling controls, KV caching, conversation state,
prompt formatting, coding-oriented evaluation flows, and inference tests.

Status: complete.

## Phase 10 — Deployment

Versioned model packaging, service entry points, health checks, observability,
containerization, security controls, and deployment documentation.

Status: complete.

## Phase 11 — Production hardening

Scale tests, failure recovery, compatibility policy, artifact provenance,
operational runbooks, and release automation based on measured deployment needs.

Status: complete.

The planned architecture roadmap is complete. Additional phases are selected
from measured model quality and production behavior, not speculative scope.

## Phase 12 - End-to-end experiment execution

One atomic command composes source verification, preprocessing, tokenizer
training, model training, evaluation, checkpointing, and inference bundle
creation.

Status: complete.

## Phase 13 - Training preflight and capacity report

Validate source splits, configuration compatibility, parameter count, schedule
size, persistent training-state lower bounds, and requested-device availability
without allocating model weights.

Status: complete.
