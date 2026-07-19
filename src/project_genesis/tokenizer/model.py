"""Immutable vocabulary and byte-level BPE encoding model."""

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from project_genesis.datasets import Dataset
from project_genesis.tokenizer.config import SpecialTokens


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Stable special, byte, and learned token ID mapping."""

    special_tokens: SpecialTokens
    tokens: tuple[bytes | None, ...]

    def __post_init__(self) -> None:
        """Validate fixed special IDs and complete byte fallback."""
        special_count = len(self.special_tokens.values)
        if len(self.tokens) < special_count + 256:
            raise ValueError("vocabulary must contain all special and byte tokens")
        if any(token is not None for token in self.tokens[:special_count]):
            raise ValueError("special-token vocabulary entries must be null byte values")
        for byte_value, token in enumerate(self.tokens[special_count : special_count + 256]):
            if token != bytes((byte_value,)):
                raise ValueError("byte-token IDs must cover bytes 0 through 255 in order")
        if any(token is None or not token for token in self.tokens[special_count:]):
            raise ValueError("non-special vocabulary entries must contain bytes")

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        """Return padding token ID."""
        return 0

    @property
    def bos_id(self) -> int:
        """Return beginning-of-sequence token ID."""
        return 1

    @property
    def eos_id(self) -> int:
        """Return end-of-sequence token ID."""
        return 2

    @property
    def unk_id(self) -> int:
        """Return reserved unknown token ID."""
        return 3

    @property
    def byte_offset(self) -> int:
        """Return the first byte token ID."""
        return len(self.special_tokens.values)

    def bytes_for(self, token_id: int) -> bytes:
        """Return byte content for a non-special token ID."""
        self._validate_id(token_id)
        token = self.tokens[token_id]
        if token is None:
            raise ValueError(f"token ID is special and has no bytes: {token_id}")
        return token

    def special_for(self, token_id: int) -> str | None:
        """Return a special-token string, or ``None`` for byte-backed IDs."""
        self._validate_id(token_id)
        return (
            self.special_tokens.values[token_id]
            if token_id < len(self.special_tokens.values)
            else None
        )

    def _validate_id(self, token_id: int) -> None:
        if isinstance(token_id, bool) or not 0 <= token_id < len(self.tokens):
            raise ValueError(f"token ID is outside vocabulary: {token_id}")


@dataclass(frozen=True, slots=True)
class MergeRule:
    """One ranked byte-pair merge."""

    left: int
    right: int
    result: int


@dataclass(frozen=True, slots=True)
class ByteBPETokenizer:
    """Deterministic UTF-8 byte-level BPE tokenizer."""

    vocabulary: Vocabulary
    merges: tuple[MergeRule, ...]
    add_bos: bool = False
    add_eos: bool = False
    _merge_ranks: Mapping[tuple[int, int], tuple[int, int]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate merge references and cache immutable merge ranks."""
        ranks: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, rule in enumerate(self.merges):
            pair = (rule.left, rule.right)
            if pair in ranks:
                raise ValueError(f"duplicate merge pair: {pair}")
            try:
                expected = self.vocabulary.bytes_for(rule.left) + self.vocabulary.bytes_for(
                    rule.right
                )
                actual = self.vocabulary.bytes_for(rule.result)
            except ValueError as error:
                raise ValueError(f"invalid merge rule at rank {rank}: {error}") from error
            if actual != expected:
                raise ValueError(f"merge result bytes do not match pair at rank {rank}")
            ranks[pair] = (rank, rule.result)
        object.__setattr__(self, "_merge_ranks", MappingProxyType(ranks))

    @property
    def fingerprint(self) -> str:
        """Return a canonical SHA-256 fingerprint of vocabulary and merges."""
        encoded = json.dumps(
            self.fingerprint_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def fingerprint_payload(self) -> dict[str, object]:
        """Return canonical JSON-compatible tokenizer identity data."""
        return {
            "special_tokens": {
                "pad": self.vocabulary.special_tokens.pad,
                "bos": self.vocabulary.special_tokens.bos,
                "eos": self.vocabulary.special_tokens.eos,
                "unk": self.vocabulary.special_tokens.unk,
            },
            "tokens": [
                None if token is None else base64.b64encode(token).decode("ascii")
                for token in self.vocabulary.tokens
            ],
            "merges": [[rule.left, rule.right, rule.result] for rule in self.merges],
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
        }

    def encode(
        self,
        text: str,
        *,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ) -> tuple[int, ...]:
        """Encode arbitrary Unicode text into stable token IDs."""
        token_ids: list[int] = []
        if self.add_bos if add_bos is None else add_bos:
            token_ids.append(self.vocabulary.bos_id)
        for piece in pretokenize(text):
            token_ids.extend(self._encode_piece(piece))
        if self.add_eos if add_eos is None else add_eos:
            token_ids.append(self.vocabulary.eos_id)
        return tuple(token_ids)

    def decode(
        self,
        token_ids: tuple[int, ...],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to Unicode, optionally retaining special literals."""
        parts: list[str] = []
        byte_buffer = bytearray()

        def flush() -> None:
            if byte_buffer:
                try:
                    parts.append(bytes(byte_buffer).decode("utf-8"))
                except UnicodeDecodeError as error:
                    raise ValueError("token sequence does not form valid UTF-8") from error
                byte_buffer.clear()

        for token_id in token_ids:
            special = self.vocabulary.special_for(token_id)
            if special is None:
                byte_buffer.extend(self.vocabulary.bytes_for(token_id))
            elif not skip_special_tokens:
                flush()
                parts.append(special)
        flush()
        return "".join(parts)

    def _encode_piece(self, piece: bytes) -> tuple[int, ...]:
        symbols = [self.vocabulary.byte_offset + byte for byte in piece]
        while len(symbols) > 1:
            candidates = (
                (rank, pair, result)
                for pair in zip(symbols, symbols[1:], strict=False)
                if (merge := self._merge_ranks.get(pair)) is not None
                for rank, result in (merge,)
            )
            selected = min(candidates, default=None)
            if selected is None:
                break
            _, pair, result = selected
            symbols = _merge_pair(symbols, pair, result)
        return tuple(symbols)


def pretokenize(text: str) -> tuple[bytes, ...]:
    """Split text without loss at whitespace/non-whitespace boundaries."""
    return tuple(match.group().encode("utf-8") for match in re.finditer(r"\s+|\S+", text))


def tokenize_dataset(dataset: Dataset, tokenizer: ByteBPETokenizer) -> Dataset:
    """Return a dataset whose records contain tokenizer-generated IDs."""
    records = tuple(replace(record, token_ids=tokenizer.encode(record.text)) for record in dataset)
    return Dataset(dataset.metadata, records, dataset.schema)


def _merge_pair(
    symbols: list[int],
    pair: tuple[int, int],
    result: int,
) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            merged.append(result)
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged
