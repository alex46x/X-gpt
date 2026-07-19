"""Typed byte-level BPE tokenizer configuration."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from project_genesis.configuration import (
    ConfigurationError,
    load_yaml,
    require_mapping,
    validate_keys,
)


@dataclass(frozen=True, slots=True)
class SpecialTokens:
    """Ordered special-token strings with stable IDs."""

    pad: str
    bos: str
    eos: str
    unk: str

    def __post_init__(self) -> None:
        """Require four unique, non-empty special tokens."""
        values = self.values
        if any(not value for value in values):
            raise ValueError("special tokens must be non-empty")
        if len(values) != len(set(values)):
            raise ValueError("special tokens must be unique")

    @property
    def values(self) -> tuple[str, str, str, str]:
        """Return tokens in their stable ID order."""
        return (self.pad, self.bos, self.eos, self.unk)


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Training and default encoding policy for byte-level BPE."""

    vocab_size: int
    min_pair_frequency: int
    special_tokens: SpecialTokens
    add_bos: bool
    add_eos: bool

    def __post_init__(self) -> None:
        """Validate vocabulary bounds and merge frequency."""
        minimum = len(self.special_tokens.values) + 256
        if self.vocab_size < minimum:
            raise ValueError(f"vocab_size must be at least {minimum}")
        if self.min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be positive")

    @property
    def fingerprint(self) -> str:
        """Return a canonical SHA-256 fingerprint of the tokenizer policy."""
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_tokenizer_config(
    path: Path,
    overrides: Sequence[str] = (),
) -> TokenizerConfig:
    """Load and strictly validate tokenizer YAML configuration."""
    root = load_yaml(path, overrides)
    validate_keys(root, required={"tokenizer"}, optional=set(), location="root")
    values = require_mapping(root["tokenizer"], "tokenizer")
    validate_keys(
        values,
        required={
            "vocab_size",
            "min_pair_frequency",
            "special_tokens",
            "add_bos",
            "add_eos",
        },
        optional=set(),
        location="tokenizer",
    )
    special = require_mapping(values["special_tokens"], "tokenizer.special_tokens")
    validate_keys(
        special,
        required={"pad", "bos", "eos", "unk"},
        optional=set(),
        location="tokenizer.special_tokens",
    )
    try:
        return TokenizerConfig(
            vocab_size=_integer(values["vocab_size"], "tokenizer.vocab_size"),
            min_pair_frequency=_integer(
                values["min_pair_frequency"],
                "tokenizer.min_pair_frequency",
            ),
            special_tokens=SpecialTokens(
                pad=_string(special["pad"], "tokenizer.special_tokens.pad"),
                bos=_string(special["bos"], "tokenizer.special_tokens.bos"),
                eos=_string(special["eos"], "tokenizer.special_tokens.eos"),
                unk=_string(special["unk"], "tokenizer.special_tokens.unk"),
            ),
            add_bos=_boolean(values["add_bos"], "tokenizer.add_bos"),
            add_eos=_boolean(values["add_eos"], "tokenizer.add_eos"),
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid tokenizer configuration: {error}") from error


def _string(value: object, location: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{location} must be a string")
    return value


def _integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be an integer")
    return value


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{location} must be a boolean")
    return value
