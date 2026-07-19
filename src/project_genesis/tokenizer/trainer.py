"""Deterministic byte-pair vocabulary training from cleaned datasets."""

from collections import Counter
from dataclasses import dataclass

from project_genesis.datasets import Dataset
from project_genesis.tokenizer.config import TokenizerConfig
from project_genesis.tokenizer.model import (
    ByteBPETokenizer,
    MergeRule,
    Vocabulary,
    pretokenize,
)


@dataclass(frozen=True, slots=True)
class TokenizerTrainingReport:
    """Corpus and vocabulary counts from one tokenizer training run."""

    documents: int
    utf8_bytes: int
    unique_pieces: int
    requested_vocab_size: int
    actual_vocab_size: int
    merges: int
    dataset_fingerprint: str
    config_fingerprint: str
    tokenizer_fingerprint: str


@dataclass(frozen=True, slots=True)
class TokenizerTrainingResult:
    """Trained tokenizer and its deterministic training report."""

    tokenizer: ByteBPETokenizer
    report: TokenizerTrainingReport


def train_tokenizer(dataset: Dataset, config: TokenizerConfig) -> TokenizerTrainingResult:
    """Train byte-level BPE merges from cleaned dataset text."""
    special_count = len(config.special_tokens.values)
    tokens: list[bytes | None] = [None] * special_count + [
        bytes((byte_value,)) for byte_value in range(256)
    ]
    token_ids = {token: token_id for token_id, token in enumerate(tokens) if token is not None}
    piece_counts: Counter[tuple[int, ...]] = Counter()
    total_bytes = 0
    for record in dataset:
        encoded = record.text.encode("utf-8")
        total_bytes += len(encoded)
        for byte_piece in pretokenize(record.text):
            piece_counts[tuple(special_count + byte for byte in byte_piece)] += 1

    unique_pieces = len(piece_counts)
    merges: list[MergeRule] = []
    # ponytail: exact full-corpus pair recount; use sharded counts when profiling requires it.
    while len(tokens) < config.vocab_size:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for symbol_piece, frequency in piece_counts.items():
            for pair in zip(symbol_piece, symbol_piece[1:], strict=False):
                pair_counts[pair] += frequency
        if not pair_counts:
            break
        pair, frequency = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        if frequency < config.min_pair_frequency:
            break

        left, right = pair
        combined = _token_bytes(tokens, left) + _token_bytes(tokens, right)
        result = token_ids.get(combined)
        if result is None:
            result = len(tokens)
            tokens.append(combined)
            token_ids[combined] = result
        merges.append(MergeRule(left, right, result))

        updated: Counter[tuple[int, ...]] = Counter()
        for symbol_piece, count in piece_counts.items():
            updated[_merge_piece(symbol_piece, pair, result)] += count
        piece_counts = updated

    tokenizer = ByteBPETokenizer(
        vocabulary=Vocabulary(config.special_tokens, tuple(tokens)),
        merges=tuple(merges),
        add_bos=config.add_bos,
        add_eos=config.add_eos,
    )
    report = TokenizerTrainingReport(
        documents=len(dataset),
        utf8_bytes=total_bytes,
        unique_pieces=unique_pieces,
        requested_vocab_size=config.vocab_size,
        actual_vocab_size=len(tokens),
        merges=len(merges),
        dataset_fingerprint=dataset.fingerprint,
        config_fingerprint=config.fingerprint,
        tokenizer_fingerprint=tokenizer.fingerprint,
    )
    return TokenizerTrainingResult(tokenizer, report)


def _token_bytes(tokens: list[bytes | None], token_id: int) -> bytes:
    token = tokens[token_id]
    if token is None:
        raise ValueError("BPE cannot merge special tokens")
    return token


def _merge_piece(
    piece: tuple[int, ...],
    pair: tuple[int, int],
    result: int,
) -> tuple[int, ...]:
    merged: list[int] = []
    index = 0
    while index < len(piece):
        if index + 1 < len(piece) and (piece[index], piece[index + 1]) == pair:
            merged.append(result)
            index += 2
        else:
            merged.append(piece[index])
            index += 1
    return tuple(merged)
