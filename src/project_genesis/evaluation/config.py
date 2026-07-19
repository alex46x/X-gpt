"""Typed evaluation and regression configuration."""

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
class RegressionThresholds:
    """Allowed degradation from a stored evaluation baseline."""

    max_loss_increase: float
    max_accuracy_drop: float
    min_throughput_ratio: float

    def __post_init__(self) -> None:
        """Validate non-negative deltas and the throughput ratio."""
        if self.max_loss_increase < 0 or self.max_accuracy_drop < 0:
            raise ValueError("regression deltas cannot be negative")
        if not 0 < self.min_throughput_ratio <= 1:
            raise ValueError("min_throughput_ratio must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Evaluation sample bound and regression policy."""

    max_batches: int
    regression: RegressionThresholds

    def __post_init__(self) -> None:
        """Require a positive evaluation bound."""
        if self.max_batches <= 0:
            raise ValueError("max_batches must be positive")


def load_evaluation_config(
    path: Path,
    overrides: Sequence[str] = (),
) -> EvaluationConfig:
    """Load and strictly validate evaluation YAML configuration."""
    root = load_yaml(path, overrides)
    validate_keys(root, required={"evaluation"}, optional=set(), location="root")
    values = require_mapping(root["evaluation"], "evaluation")
    validate_keys(
        values,
        required={"max_batches", "regression"},
        optional=set(),
        location="evaluation",
    )
    regression = require_mapping(values["regression"], "evaluation.regression")
    validate_keys(
        regression,
        required={
            "max_loss_increase",
            "max_accuracy_drop",
            "min_throughput_ratio",
        },
        optional=set(),
        location="evaluation.regression",
    )
    try:
        return EvaluationConfig(
            max_batches=_integer(values["max_batches"], "evaluation.max_batches"),
            regression=RegressionThresholds(
                max_loss_increase=_number(
                    regression["max_loss_increase"],
                    "evaluation.regression.max_loss_increase",
                ),
                max_accuracy_drop=_number(
                    regression["max_accuracy_drop"],
                    "evaluation.regression.max_accuracy_drop",
                ),
                min_throughput_ratio=_number(
                    regression["min_throughput_ratio"],
                    "evaluation.regression.min_throughput_ratio",
                ),
            ),
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid evaluation configuration: {error}") from error


def _integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be an integer")
    return value


def _number(value: object, location: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be a number")
    return float(value)
