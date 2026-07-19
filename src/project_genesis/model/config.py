"""Typed configuration for decoder model primitives."""

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
class ModelConfig:
    """Dimensions and numerical policy shared by model primitives."""

    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    bias: bool
    layer_norm_epsilon: float
    n_layers: int = 12
    initializer_range: float = 0.02
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """Validate dimensions and probabilities."""
        for name in (
            "vocab_size",
            "context_length",
            "d_model",
            "n_heads",
            "d_ff",
            "n_layers",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")

    @property
    def head_dim(self) -> int:
        """Return per-head channel width."""
        return self.d_model // self.n_heads


def load_model_config(
    path: Path,
    overrides: Sequence[str] = (),
) -> ModelConfig:
    """Load and strictly validate model YAML configuration."""
    root = load_yaml(path, overrides)
    validate_keys(root, required={"model"}, optional=set(), location="root")
    values = require_mapping(root["model"], "model")
    fields = {
        "vocab_size",
        "context_length",
        "d_model",
        "n_heads",
        "d_ff",
        "dropout",
        "bias",
        "layer_norm_epsilon",
        "n_layers",
        "initializer_range",
        "tie_embeddings",
    }
    validate_keys(values, required=fields, optional=set(), location="model")
    try:
        return ModelConfig(
            vocab_size=_integer(values["vocab_size"], "model.vocab_size"),
            context_length=_integer(values["context_length"], "model.context_length"),
            d_model=_integer(values["d_model"], "model.d_model"),
            n_heads=_integer(values["n_heads"], "model.n_heads"),
            d_ff=_integer(values["d_ff"], "model.d_ff"),
            dropout=_number(values["dropout"], "model.dropout"),
            bias=_boolean(values["bias"], "model.bias"),
            layer_norm_epsilon=_number(
                values["layer_norm_epsilon"],
                "model.layer_norm_epsilon",
            ),
            n_layers=_integer(values["n_layers"], "model.n_layers"),
            initializer_range=_number(
                values["initializer_range"],
                "model.initializer_range",
            ),
            tie_embeddings=_boolean(values["tie_embeddings"], "model.tie_embeddings"),
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid model configuration: {error}") from error


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
