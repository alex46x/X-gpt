# Phase 6 — Decoder-only architecture

## Phase overview

Phase 6 composes the independently tested model primitives into a complete
decoder-only network. The public model accepts batch-first token IDs and returns
unnormalized next-token logits. Training loss, optimization, checkpointing, and
generation remain outside this phase.

## Architecture

Each transformer block uses a pre-normalization layout:

```text
input → layer norm → causal self-attention → residual add
      → layer norm → feed-forward          → residual add
```

The decoder composes the full token-to-logits path:

```text
token IDs
  → token embedding + learned position embedding
  → dropout
  → N transformer blocks
  → final layer norm
  → vocabulary projection
  → logits
```

The causal mask belongs to each attention module and is not persisted in model
state. The decoder therefore has no parallel masking policy to keep synchronized.

## Configuration and interfaces

`ModelConfig` adds three architecture fields:

- `n_layers`: positive number of transformer blocks.
- `initializer_range`: positive standard deviation for embeddings and ordinary
  linear projections.
- `tie_embeddings`: whether the output projection shares the token-embedding
  parameter.

`TransformerBlock.forward` accepts and returns
`(batch, sequence, d_model)`. `GPTDecoder.forward` accepts integer
`(batch, sequence)` token IDs and returns
`(batch, sequence, vocab_size)` logits. Empty sequences and sequences beyond the
configured context window are rejected.

`parameter_count` uses PyTorch's unique parameter traversal, so tied weights are
counted once. Its optional `trainable_only` mode excludes frozen parameters.

## Initialization decisions

Linear and embedding weights use a zero-mean normal distribution with the
configured standard deviation. Linear biases are zero. Attention and
feed-forward output projections use a standard deviation divided by
`sqrt(2 * n_layers)` to account for residual accumulation. Layer-normalization
scale and bias retain their natural one and zero initialization.

Tying occurs after initialization so the vocabulary projection and token lookup
refer to the same parameter object. Untied output weights remain independently
initialized.

## Dependency boundary

```text
configuration → model primitives → TransformerBlock → GPTDecoder
```

The model has no dependency on dataset, preprocessing, tokenizer, training,
evaluation, inference, chat, benchmark, or deployment modules. Vocabulary size
is the configuration contract between a trained tokenizer and the model; runtime
compatibility validation belongs at their future composition boundary.

## Testing and risks

Tests cover shape preservation, gradients, causal isolation across the complete
decoder, context limits, weight tying, parameter accounting, zero biases, and
scaled residual initialization. Dropout is disabled in behavioral tests to make
causality and numerical assertions deterministic.

The manual attention implementation remains quadratic in sequence length.
Optimized attention kernels, activation checkpointing, distributed execution,
mixed precision, language-model loss, checkpoint formats, KV caches, and
generation are intentionally deferred until their consuming phases can provide
measured requirements.

## Acceptance commands

```powershell
uv sync --locked --group dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest -W error
uv build
```
