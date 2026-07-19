"""Validated atomic JSON serialization for trained tokenizers."""

import base64
import binascii
import json
from pathlib import Path

from project_genesis.tokenizer.config import SpecialTokens
from project_genesis.tokenizer.model import ByteBPETokenizer, MergeRule, Vocabulary
from project_genesis.utilities import atomic_write_text


def save_tokenizer(tokenizer: ByteBPETokenizer, path: Path) -> None:
    """Atomically save a tokenizer with its integrity fingerprint."""
    document = {
        "schema_version": "1.0.0",
        "tokenizer": tokenizer.fingerprint_payload(),
        "fingerprint": tokenizer.fingerprint,
    }
    atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n")


def load_tokenizer(path: Path) -> ByteBPETokenizer:
    """Load and fully validate a serialized tokenizer."""
    try:
        with path.expanduser().resolve().open(encoding="utf-8") as stream:
            decoded: object = json.load(stream)
        document = _mapping(decoded, "document")
        _exact_keys(document, {"schema_version", "tokenizer", "fingerprint"}, "document")
        if _string(document, "schema_version") != "1.0.0":
            raise ValueError("unsupported tokenizer schema version")
        payload = _mapping(document.get("tokenizer"), "tokenizer")
        _exact_keys(
            payload,
            {"special_tokens", "tokens", "merges", "add_bos", "add_eos"},
            "tokenizer",
        )
        special = _mapping(payload.get("special_tokens"), "special_tokens")
        _exact_keys(special, {"pad", "bos", "eos", "unk"}, "special_tokens")
        tokenizer = ByteBPETokenizer(
            vocabulary=Vocabulary(
                SpecialTokens(
                    pad=_string(special, "pad"),
                    bos=_string(special, "bos"),
                    eos=_string(special, "eos"),
                    unk=_string(special, "unk"),
                ),
                _tokens(payload.get("tokens")),
            ),
            merges=_merges(payload.get("merges")),
            add_bos=_boolean(payload, "add_bos"),
            add_eos=_boolean(payload, "add_eos"),
        )
        if _string(document, "fingerprint") != tokenizer.fingerprint:
            raise ValueError("tokenizer fingerprint does not match its contents")
        return tokenizer
    except (OSError, json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise ValueError(f"Unable to load tokenizer {path}: {error}") from error


def _tokens(value: object) -> tuple[bytes | None, ...]:
    if not isinstance(value, list):
        raise ValueError("tokens must be a list")
    tokens: list[bytes | None] = []
    for index, token in enumerate(value):
        if token is None:
            tokens.append(None)
        elif isinstance(token, str):
            try:
                tokens.append(base64.b64decode(token, validate=True))
            except (binascii.Error, ValueError) as error:
                raise ValueError(f"token {index} is not valid base64") from error
        else:
            raise ValueError(f"token {index} must be base64 text or null")
    return tuple(tokens)


def _merges(value: object) -> tuple[MergeRule, ...]:
    if not isinstance(value, list):
        raise ValueError("merges must be a list")
    merges: list[MergeRule] = []
    for index, merge in enumerate(value):
        if (
            not isinstance(merge, list)
            or len(merge) != 3
            or any(not isinstance(item, int) or isinstance(item, bool) for item in merge)
        ):
            raise ValueError(f"merge {index} must contain three integer IDs")
        merges.append(MergeRule(merge[0], merge[1], merge[2]))
    return tuple(merges)


def _mapping(value: object, location: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return value


def _exact_keys(
    values: dict[object, object],
    expected: set[str],
    location: str,
) -> None:
    keys = set(values)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(str(key) for key in keys - expected)
        raise ValueError(f"{location} keys are invalid; missing={missing}, unknown={unknown}")


def _string(values: dict[object, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _boolean(values: dict[object, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value
