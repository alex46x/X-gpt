# Phase 5: Model Primitives

## Goal

Phase 5 implements the independently testable PyTorch layers required by a
decoder-only transformer block: token and position embeddings, layer
normalization, feed-forward transformation, causal multi-head self-attention,
and residual addition.

It does not compose transformer blocks, build a decoder, compute language-model
loss, initialize a complete model, cache keys/values, or train parameters.

## Architecture

```text
configuration
     ↓
model.config
     ├── embeddings
     ├── normalization
     ├── feed_forward
     ├── attention
     └── residual
```

All layers accept batch-first tensors. Hidden states use
`(batch, sequence, d_model)`; token IDs use `(batch, sequence)`.

## Configuration

The external YAML contract defines vocabulary size, context length, model width,
head count, feed-forward width, dropout, projection bias, and layer-normalization
epsilon. Validation requires positive dimensions, `d_model` divisible by
`n_heads`, dropout in `[0, 1)`, and positive epsilon.

The default vocabulary size matches the Phase 4 tokenizer default. The actual
trained tokenizer vocabulary and model configuration must be checked together
when Phase 6 composes the decoder.

## Embeddings

`TokenEmbedding` wraps a learned lookup table and accepts only rank-two integer
IDs. `LearnedPositionEmbedding` creates positions on the input device and rejects
sequences beyond the configured context. It returns shape
`(1, sequence, d_model)` so PyTorch broadcasts positions across the batch.

## Layer normalization and feed-forward

Layer normalization is implemented directly from per-token mean and population
variance with trainable scale and optional bias. Half and bfloat16 statistics
accumulate in float32, then return to the input dtype.

The feed-forward layer applies:

```text
Linear(d_model, d_ff) → exact GELU → Linear(d_ff, d_model) → Dropout
```

Dimensions and dropout are external configuration, not hidden constants.

## Causal self-attention

Attention performs explicit combined QKV projection, head splitting, scaled
query/key products, lower-triangular masking, softmax, attention dropout, value
aggregation, head concatenation, output projection, and output dropout.

The mask is a non-persistent boolean buffer, so it follows the module device but
does not inflate checkpoints. Half and bfloat16 softmax uses float32 before
returning to the value dtype.

The current implementation is intentionally quadratic and transparent.
PyTorch SDPA or FlashAttention becomes a kernel-level replacement when profiling
shows the manual implementation is the bottleneck; the public tensor contract
does not need to change.

## Residual connection

Residual addition remains a function because the operation has no state. It
rejects broadcasting, dtype promotion, and cross-device addition before applying
`inputs + update`.

## Compute and dependencies

The locked development and CI environment uses PyTorch 2.13 CPU wheels from
PyTorch's explicit CPU index. NumPy is explicit because PyTorch imports its NumPy
bridge during initialization.

CUDA remains the training target. A developer with a compatible driver may
replace the locked CPU wheel inside the synchronized environment:

```console
uv pip install --reinstall torch --torch-backend=auto
```

Running `uv sync` restores the reproducible CPU lock. CUDA CI and a fixed CUDA
wheel are deferred until the training phase establishes its actual accelerator
and driver requirements.

## Validation

Tests cover configuration failures, tensor shapes, gradients, context bounds,
custom/native LayerNorm equivalence, half-precision output, deterministic eval
dropout behavior, causal invariance, non-persistent masks, and residual
broadcast rejection.
