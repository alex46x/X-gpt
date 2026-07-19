# Phase 4: Vocabulary and Tokenizer

## Goal

Phase 4 trains a custom reversible tokenizer directly from cleaned Project
Genesis datasets. It owns vocabulary construction, stable token IDs, byte-pair
merge training, encoding, decoding, persistence, and quality measurements.

No pretrained vocabulary, tokenizer package, PyTorch model, embedding, loss, or
training-engine behavior is used.

## Architecture

```text
configuration + datasets
          ↓
tokenizer.config
          ↓
tokenizer.trainer → tokenizer.model
                          ├── tokenizer.storage
                          ├── tokenizer.evaluation
                          └── tokenized Dataset records
```

- `config.py` validates the external training and special-token policy.
- `model.py` owns immutable vocabulary, merge rules, encoding, and decoding.
- `trainer.py` builds deterministic byte-pair merges from cleaned text.
- `storage.py` validates and atomically persists complete tokenizer state.
- `evaluation.py` measures exact round trips and bytes represented per token.

## Token ID contract

IDs are stable:

```text
0       padding
1       beginning of sequence
2       end of sequence
3       reserved unknown token
4–259   raw bytes 0–255
260+    learned byte-pair tokens
```

The four special strings are configurable and must be non-empty and unique.
Byte fallback covers every UTF-8 input, so encoding never emits the unknown ID.
Special literals appearing in ordinary text are encoded as bytes; callers add
special IDs explicitly through encoding policy.

## Training algorithm

Text is split losslessly into whitespace and non-whitespace pieces. This retains
source-code indentation and exact decoding while preventing merges across
arbitrary word boundaries.

Training counts adjacent token pairs with corpus frequencies, chooses the highest
frequency, and resolves ties by the lowest token-ID pair. Each selected pair is
merged left-to-right across all unique pieces. Training stops at the requested
vocabulary size, below the configured frequency threshold, or when no pairs
remain.

The implementation uses exact in-memory piece and pair counts. This is simple,
deterministic, and appropriate for the current local dataset contract. Sharded
or external pair counting should replace it only when profiling shows corpus
scale exceeds available memory or acceptable training time.

## Encoding and dataset integration

Encoding starts from UTF-8 byte IDs and repeatedly applies the highest-ranked
available merge within each lossless piece. Decoding concatenates token bytes
and performs strict UTF-8 decoding. Invalid IDs and byte sequences fail rather
than silently replacing text.

`tokenize_dataset` creates new immutable records with `token_ids` populated.
Original text, checksums, document IDs, provenance, metadata, labels, and
embeddings remain unchanged.

## Reproducibility and storage

Tokenizer fingerprints cover:

- Special-token strings.
- Every byte and learned vocabulary entry.
- Ranked merge rules.
- Default BOS/EOS behavior.

Training reports bind dataset, configuration, and tokenizer fingerprints with
corpus and vocabulary counts. Serialized JSON uses base64 for token bytes,
includes a schema version and fingerprint, rejects unknown fields, and is
written atomically.

## Quality and limits

Quality evaluation reports document count, UTF-8 bytes, encoded token count,
bytes per token, and round-trip failures. Tests cover multilingual unseen text,
emoji, whitespace, special literals, stable IDs, deterministic training,
serialization tampering, and dataset integration.

The tokenizer intentionally omits regex language-specific pre-tokenization,
normalization inside the tokenizer, dropout, approximate BPE, and distributed
training. Text normalization belongs to Phase 3; additional algorithms require
measured quality evidence rather than speculative interfaces.
