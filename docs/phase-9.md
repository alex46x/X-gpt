# Phase 9 — Inference and chat

## Phase overview

Phase 9 adds bounded autoregressive decoding, KV caching, sampling controls,
immutable conversation state, deterministic prompt assembly, generated replies,
and exact-match completion benchmarks. It composes the existing model and
tokenizer without introducing a server or deployment protocol.

## Architecture

```text
prompt token IDs
  → full prompt forward + per-layer KV cache
  → final-position logits
  → repetition penalty
  → temperature / top-k / top-p
  → greedy choice or multinomial sample
  → cached one-token forwards
  → stop, length, or context finish

Conversation + user message
  → role-delimited prompt
  → tokenizer
  → generation
  → tokenizer decode
  → new immutable Conversation
```

Dependency direction is `model + tokenizer + configuration → inference → chat`.
Completion benchmarks depend on inference; model and inference never import
benchmark code.

## KV-cache contract

Each attention layer exposes an immutable `(key, value)` tensor pair with shape
`(batch, heads, cached_sequence, head_dimension)`. The decoder cache contains one
pair per transformer block. Cache batch, head, channel, dtype, device, layer
count, and sequence lengths are validated.

Cached attention slices the same registered causal mask used by full forward.
Learned positional embeddings receive the cached sequence length as their
offset. Numerical tests compare split cached outputs with a complete forward
through both attention and the full decoder.

The original `model(token_ids)` interface remains unchanged for training and
evaluation. `forward_cached` is opt-in. Cache tensors currently grow by
concatenation; fixed-capacity allocation should replace this only if inference
profiling shows copying is material.

## Generation configuration

`GenerationConfig` validates:

- Positive maximum new-token count.
- Zero-or-positive temperature, where zero selects greedy decoding.
- Optional top-k filtering.
- Top-p nucleus probability.
- Repetition penalty of at least one.
- Unique non-negative stop-token IDs.
- KV-cache enablement.

The strict default is `configs/inference/default.yaml`. A caller may inject a
device-compatible `torch.Generator` for reproducible stochastic sampling.

## Generation behavior

Generation supports one prompt sequence per call. Prompt and stop IDs are
validated against the model vocabulary. The prompt may not exceed the model
context, and generation never truncates or evicts existing tokens silently.

`GenerationResult` contains only the generated suffix and a finish reason:
`stop`, `length`, or `context`. Stop tokens remain in the token result so callers
can audit the exact model output; tokenizer decoding skips configured special
tokens by its existing policy.

The model's training/evaluation mode is restored even when generation fails.
Inference uses `torch.inference_mode`.

## Chat

`Conversation` and `Message` are immutable. A system message is optional and may
appear only first; user and assistant messages must then alternate.

Prompt formatting uses visible role delimiters:

```text
<|system|>
...
<|user|>
...
<|assistant|>
```

`generate_reply` appends the user message, encodes the complete prompt, generates
and decodes a suffix, then returns a new conversation with the assistant reply.
The original conversation is unchanged. Empty decoded replies are rejected.

## Coding completion evaluation

`run_completion_benchmark` accepts named and categorized prompt/expected-token
cases, allowing coding cases without a coding-specific model abstraction. It
records generated suffixes, finish reasons, exact matches, aggregate accuracy,
and a SHA-256 fingerprint of the ordered suite.

Code execution is not performed. Running untrusted generated programs requires a
separate hardened sandbox and belongs to deployment or production hardening.

## Testing

Tests cover cached/full numerical parity, cache growth, strict configuration,
cached versus uncached greedy equality, stop and context termination, repetition
penalties, seeded top-k/top-p sampling, invalid IDs, role ordering, prompt
formatting, tokenizer/model chat composition, and coding completion scoring.

## Acceptance commands

```powershell
uv sync --locked --group dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest -W error
uv build
```
