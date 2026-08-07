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


class InitMode(StrEnum):
    """Select random initialization or checkpoint continuation."""

    RANDOM = "random"
    RESUME = "resume"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization, scheduling, checkpoint, and resume policy."""

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
    checkpoint_interval_steps: int
    evaluation_interval_steps: int
    log_interval_steps: int
    keep_last_checkpoints: int
    init_mode: InitMode = InitMode.RANDOM
    resume_from: Path | None = None
    save_checkpoint_every: int | None = None
    save_latest: bool = True
    save_best: bool = True
    output_directory: Path | None = None

    def __post_init__(self) -> None:
        """Validate training bounds and checkpoint settings."""
        for name in (
            "batch_size",
            "sequence_length",
            "max_steps",
            "gradient_accumulation_steps",
            "checkpoint_interval_steps",
            "evaluation_interval_steps",
            "log_interval_steps",
            "keep_last_checkpoints",
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
        if self.save_checkpoint_every is not None and self.save_checkpoint_every <= 0:
            raise ValueError("save_checkpoint_every must be positive")
        if self.init_mode is InitMode.RESUME and self.resume_from is None:
            raise ValueError("resume_from is required when init_mode is resume")


def load_training_config(
    path: Path,
    overrides: Sequence[str] = (),
) -> TrainingConfig:
    """Load and strictly validate training YAML."""
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
        "checkpoint_interval_steps",
        "evaluation_interval_steps",
        "log_interval_steps",
        "keep_last_checkpoints",
    }
    optional = {
        "init_mode",
        "resume_from",
        "save_checkpoint_every",
        "save_latest",
        "save_best",
        "output_directory",
    }
    validate_keys(values, required=fields, optional=optional, location="training")
    try:
        return TrainingConfig(
            batch_size=_integer(values["batch_size"], "training.batch_size"),
            sequence_length=_integer(values["sequence_length"], "training.sequence_length"),
            learning_rate=_number(values["learning_rate"], "training.learning_rate"),
            weight_decay=_number(values["weight_decay"], "training.weight_decay"),
            beta1=_number(values["beta1"], "training.beta1"),
            beta2=_number(values["beta2"], "training.beta2"),
            epsilon=_number(values["epsilon"], "training.epsilon"),
            warmup_steps=_integer(values["warmup_steps"], "training.warmup_steps"),
            max_steps=_integer(values["max_steps"], "training.max_steps"),
            min_learning_rate_ratio=_number(
                values["min_learning_rate_ratio"], "training.min_learning_rate_ratio"
            ),
            gradient_accumulation_steps=_integer(
                values["gradient_accumulation_steps"], "training.gradient_accumulation_steps"
            ),
            max_gradient_norm=_number(values["max_gradient_norm"], "training.max_gradient_norm"),
            precision=Precision(_string(values["precision"], "training.precision")),
            seed=_integer(values["seed"], "training.seed"),
            checkpoint_interval_steps=_integer(
                values["checkpoint_interval_steps"], "training.checkpoint_interval_steps"
            ),
            evaluation_interval_steps=_integer(
                values["evaluation_interval_steps"], "training.evaluation_interval_steps"
            ),
            log_interval_steps=_integer(
                values["log_interval_steps"], "training.log_interval_steps"
            ),
            keep_last_checkpoints=_integer(
                values["keep_last_checkpoints"], "training.keep_last_checkpoints"
            ),
            init_mode=InitMode(_string(values.get("init_mode", "random"), "training.init_mode")),
            resume_from=_optional_path(values.get("resume_from"), "training.resume_from"),
            save_checkpoint_every=_optional_integer(
                values.get("save_checkpoint_every"), "training.save_checkpoint_every"
            ),
            save_latest=_boolean(values.get("save_latest", True), "training.save_latest"),
            save_best=_boolean(values.get("save_best", True), "training.save_best"),
            output_directory=_optional_path(
                values.get("output_directory"), "training.output_directory"
            ),
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid training configuration: {error}") from error


def _integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be an integer")
    return value


def _optional_integer(value: object, location: str) -> int | None:
    return None if value is None else _integer(value, location)


def _number(value: object, location: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be a number")
    return float(value)


def _string(value: object, location: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{location} must be a string")
    return value


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{location} must be a boolean")
    return value


def _optional_path(value: object, location: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a non-empty path")
    return Path(value)
