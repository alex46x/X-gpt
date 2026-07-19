"""Environment and path configuration models shared by Project Genesis."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from project_genesis.configuration.loader import ConfigurationError, ConfigValue

ENVIRONMENT_VARIABLE = "PROJECT_GENESIS_ENV"


class RuntimeEnvironment(StrEnum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Absolute locations for dataset and generated artifacts."""

    data: Path
    cache: Path
    artifacts: Path

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, ConfigValue],
        *,
        config_file: Path,
    ) -> "ProjectPaths":
        """Validate and resolve a paths configuration mapping."""
        required = {"data", "cache", "artifacts"}
        unknown = set(values) - required
        missing = required - set(values)
        if unknown or missing:
            raise ConfigurationError(
                f"paths keys are invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        resolved: dict[str, Path] = {}
        for key in sorted(required):
            value = values[key]
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"paths.{key} must be a non-empty string")
            resolved[key] = resolve_config_path(value, config_file)
        return cls(**resolved)


def detect_environment(
    configured: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> RuntimeEnvironment:
    """Detect the runtime environment, preferring the process environment."""
    variables = os.environ if environ is None else environ
    value = variables.get(ENVIRONMENT_VARIABLE, configured or RuntimeEnvironment.DEVELOPMENT)
    try:
        return RuntimeEnvironment(value)
    except ValueError as error:
        choices = ", ".join(environment.value for environment in RuntimeEnvironment)
        raise ConfigurationError(
            f"{ENVIRONMENT_VARIABLE} must be one of: {choices}; received {value!r}"
        ) from error


def resolve_config_path(value: str, config_file: Path) -> Path:
    """Resolve a path relative to the declaring configuration file."""
    path = Path(value).expanduser()
    base = config_file.expanduser().resolve().parent
    return (path if path.is_absolute() else base / path).resolve()
