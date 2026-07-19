"""Safe YAML loading with strict dotted-key overrides."""

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import yaml

type ConfigScalar = str | int | float | bool | None
type ConfigValue = ConfigScalar | list[ConfigValue] | dict[str, ConfigValue]
type ConfigMapping = dict[str, ConfigValue]


class ConfigurationError(ValueError):
    """Raised when configuration input is missing, unsafe, or invalid."""


def load_yaml(path: Path, overrides: Sequence[str] = ()) -> ConfigMapping:
    """Load a YAML mapping and apply existing-key overrides.

    Args:
        path: YAML file to load.
        overrides: Dotted assignments such as ``dataset.version=2.0.0``.

    Returns:
        A normalized mapping containing only supported YAML value types.

    Raises:
        ConfigurationError: If the file or any override is invalid.
    """
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError(f"Configuration must be YAML: {path}")
    if not resolved.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    try:
        with resolved.open(encoding="utf-8") as stream:
            loaded: object = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Unable to load configuration {path}: {error}") from error

    if loaded is None:
        data: ConfigMapping = {}
    else:
        normalized = _normalize(loaded, "configuration")
        if not isinstance(normalized, dict):
            raise ConfigurationError("Configuration root must be a mapping")
        data = normalized

    result = deepcopy(data)
    for override in overrides:
        _apply_override(result, override)
    return result


def _normalize(value: object, location: str) -> ConfigValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_normalize(item, f"{location}[]") for item in value]
    if isinstance(value, dict):
        normalized: ConfigMapping = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ConfigurationError(f"{location} contains a non-string or empty key")
            normalized[key] = _normalize(item, f"{location}.{key}")
        return normalized
    raise ConfigurationError(f"{location} contains unsupported value {type(value).__name__}")


def _apply_override(config: ConfigMapping, assignment: str) -> None:
    key, separator, raw_value = assignment.partition("=")
    parts = key.split(".")
    if not separator or not raw_value or any(not part for part in parts):
        raise ConfigurationError(f"Invalid override {assignment!r}; expected dotted.key=value")

    target = config
    for part in parts[:-1]:
        value = target.get(part)
        if not isinstance(value, dict):
            raise ConfigurationError(f"Override path does not exist: {key}")
        target = value

    leaf = parts[-1]
    if leaf not in target:
        raise ConfigurationError(f"Override key does not exist: {key}")
    try:
        parsed: object = yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid override value for {key}: {error}") from error
    target[leaf] = _normalize(parsed, f"override {key}")
