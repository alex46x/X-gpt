"""Typed autoregressive generation configuration."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from project_genesis.configuration import (
    ConfigurationError,
    load_yaml,
    require_mapping,
    validate_keys,
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Length, sampling, repetition, stopping, and cache policy."""

    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    stop_token_ids: tuple[int, ...]
    use_cache: bool

    def __post_init__(self) -> None:
        """Validate generation bounds and stop-token identity."""
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be at least one")
        if len(self.stop_token_ids) != len(set(self.stop_token_ids)):
            raise ValueError("stop_token_ids cannot contain duplicates")
        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError("stop_token_ids cannot contain negative values")


def load_generation_config(
    path: Path,
    overrides: Sequence[str] = (),
) -> GenerationConfig:
    """Load and strictly validate inference YAML configuration."""
    root = load_yaml(path, overrides)
    validate_keys(root, required={"inference"}, optional=set(), location="root")
    values = require_mapping(root["inference"], "inference")
    fields = {
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "stop_token_ids",
        "use_cache",
    }
    validate_keys(values, required=fields, optional=set(), location="inference")
    stop_values = values["stop_token_ids"]
    if not isinstance(stop_values, list):
        raise ConfigurationError("inference.stop_token_ids must be a list")
    try:
        return GenerationConfig(
            max_new_tokens=_integer(
                values["max_new_tokens"],
                "inference.max_new_tokens",
            ),
            temperature=_number(values["temperature"], "inference.temperature"),
            top_k=_integer(values["top_k"], "inference.top_k"),
            top_p=_number(values["top_p"], "inference.top_p"),
            repetition_penalty=_number(
                values["repetition_penalty"],
                "inference.repetition_penalty",
            ),
            stop_token_ids=tuple(
                _integer(token_id, "inference.stop_token_ids[]") for token_id in stop_values
            ),
            use_cache=_boolean(values["use_cache"], "inference.use_cache"),
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid inference configuration: {error}") from error


def _integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be an integer")
    return value


def _number(value: object, location: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be a number")
    return float(value)


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{location} must be a boolean")
    return value
