"""Typed preprocessing policy loaded from strict YAML."""

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from project_genesis.configuration import (
    ConfigurationError,
    load_yaml,
    require_mapping,
    validate_keys,
)

type NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


class ErrorPolicy(StrEnum):
    """Behavior when a local source cannot be parsed."""

    RAISE = "raise"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Deterministic text normalization, filtering, and deduplication policy."""

    unicode_normalization: NormalizationForm
    normalize_newlines: bool
    strip_control_characters: bool
    collapse_whitespace: bool
    trim: bool
    min_characters: int
    max_characters: int
    allowed_languages: tuple[str, ...]
    deduplicate: bool
    on_error: ErrorPolicy

    def __post_init__(self) -> None:
        """Validate policy bounds and normalization form."""
        try:
            unicodedata.normalize(self.unicode_normalization, "")
        except ValueError as error:
            raise ValueError(
                f"unsupported Unicode normalization form: {self.unicode_normalization}"
            ) from error
        if self.min_characters < 1:
            raise ValueError("min_characters must be positive")
        if self.max_characters < self.min_characters:
            raise ValueError("max_characters must be at least min_characters")
        if len(self.allowed_languages) != len(set(self.allowed_languages)):
            raise ValueError("allowed_languages cannot contain duplicates")
        if any(not language.strip() for language in self.allowed_languages):
            raise ValueError("allowed_languages must contain non-empty values")

    @property
    def fingerprint(self) -> str:
        """Return the canonical SHA-256 fingerprint of this policy."""
        values = asdict(self)
        values["on_error"] = self.on_error.value
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_preprocessing_config(
    path: Path,
    overrides: Sequence[str] = (),
) -> PreprocessingConfig:
    """Load and validate a preprocessing YAML policy."""
    root = load_yaml(path, overrides)
    validate_keys(root, required={"preprocessing"}, optional=set(), location="root")
    values = require_mapping(root["preprocessing"], "preprocessing")
    fields = {
        "unicode_normalization",
        "normalize_newlines",
        "strip_control_characters",
        "collapse_whitespace",
        "trim",
        "min_characters",
        "max_characters",
        "allowed_languages",
        "deduplicate",
        "on_error",
    }
    validate_keys(values, required=fields, optional=set(), location="preprocessing")

    allowed_languages = values["allowed_languages"]
    if not isinstance(allowed_languages, list) or not all(
        isinstance(language, str) for language in allowed_languages
    ):
        raise ConfigurationError("allowed_languages must be a list of strings")
    try:
        return PreprocessingConfig(
            unicode_normalization=cast(
                NormalizationForm,
                _string(values["unicode_normalization"], "unicode_normalization"),
            ),
            normalize_newlines=_boolean(values["normalize_newlines"], "normalize_newlines"),
            strip_control_characters=_boolean(
                values["strip_control_characters"], "strip_control_characters"
            ),
            collapse_whitespace=_boolean(values["collapse_whitespace"], "collapse_whitespace"),
            trim=_boolean(values["trim"], "trim"),
            min_characters=_integer(values["min_characters"], "min_characters"),
            max_characters=_integer(values["max_characters"], "max_characters"),
            allowed_languages=tuple(
                _string(language, "allowed_languages[]") for language in allowed_languages
            ),
            deduplicate=_boolean(values["deduplicate"], "deduplicate"),
            on_error=ErrorPolicy(_string(values["on_error"], "on_error")),
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid preprocessing configuration: {error}") from error


def _string(value: object, location: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{location} must be a string")
    return value


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{location} must be a boolean")
    return value


def _integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be an integer")
    return value
