"""Typed training configuration."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from project_genesis.configuration import (
    ConfigurationError,
    load_yaml,
    require_mapping,
    validate_keys,
)


class Precision(StrEnum):
    """Supported training compute precisions."""

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization, scheduling, and numerical training policy."""

    batch_size: int
    sequence_length: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    warmup_steps: int
    max_steps: int
    min_learning_rate_ratio: float
    gradient_accumulation_steps: int
    max_gradient_norm: float
    precision: Precision
    seed: int

    def __post_init__(self) -> None:
        """Validate training bounds."""
        for name in (
            "batch_size",
            "sequence_length",
            "max_steps",
            "gradient_accumulation_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.epsilon <= 0 or self.max_gradient_norm <= 0:
            raise ValueError("learning_rate, epsilon, and max_gradient_norm must be positive")
        if self.weight_decay < 0 or self.warmup_steps < 0 or self.seed < 0:
            raise ValueError("weight_decay, warmup_steps, and seed cannot be negative")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be less than max_steps")
        if not 0 < self.min_learning_rate_ratio <= 1:
            raise ValueError("min_learning_rate_ratio must be in (0, 1]")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("optimizer betas must be in [0, 1)")


def load_training_config(
    path: Path,
    overrides: Sequence[str] = (),
) -> TrainingConfig:
    """Load and strictly validate training YAML configuration."""
    root = load_yaml(path, overrides)
    validate_keys(root, required={"training"}, optional=set(), location="root")
    values = require_mapping(root["training"], "training")
    fields = {
        "batch_size",
        "sequence_length",
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "epsilon",
        "warmup_steps",
        "max_steps",
        "min_learning_rate_ratio",
        "gradient_accumulation_steps",
        "max_gradient_norm",
        "precision",
        "seed",
    }
    validate_keys(values, required=fields, optional=set(), location="training")
    try:
        return TrainingConfig(
            batch_size=_integer(values["batch_size"], "training.batch_size"),
            sequence_length=_integer(
                values["sequence_length"],
                "training.sequence_length",
            ),
            learning_rate=_number(values["learning_rate"], "training.learning_rate"),
            weight_decay=_number(values["weight_decay"], "training.weight_decay"),
            beta1=_number(values["beta1"], "training.beta1"),
            beta2=_number(values["beta2"], "training.beta2"),
            epsilon=_number(values["epsilon"], "training.epsilon"),
            warmup_steps=_integer(values["warmup_steps"], "training.warmup_steps"),
            max_steps=_integer(values["max_steps"], "training.max_steps"),
            min_learning_rate_ratio=_number(
                values["min_learning_rate_ratio"],
                "training.min_learning_rate_ratio",
            ),
            gradient_accumulation_steps=_integer(
                values["gradient_accumulation_steps"],
                "training.gradient_accumulation_steps",
            ),
            max_gradient_norm=_number(
                values["max_gradient_norm"],
                "training.max_gradient_norm",
            ),
            precision=Precision(_string(values["precision"], "training.precision")),
            seed=_integer(values["seed"], "training.seed"),
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid training configuration: {error}") from error


def _integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be an integer")
    return value


def _number(value: object, location: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be a number")
    return float(value)


def _string(value: object, location: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{location} must be a string")
    return value
