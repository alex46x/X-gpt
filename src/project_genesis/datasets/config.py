"""Typed dataset configuration decoded from safe YAML."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from project_genesis.configuration import (
    ConfigMapping,
    ConfigurationError,
    ConfigValue,
    ProjectPaths,
    RuntimeEnvironment,
    detect_environment,
    load_yaml,
    resolve_config_path,
)
from project_genesis.datasets.records import DatasetMetadata
from project_genesis.datasets.sources import DatasetSource, DatasetSplit, SourceFormat


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Complete typed configuration required by the dataset foundation."""

    environment: RuntimeEnvironment
    paths: ProjectPaths
    metadata: DatasetMetadata
    sources: tuple[DatasetSource, ...]


def load_dataset_config(
    path: Path,
    overrides: Sequence[str] = (),
    *,
    environ: Mapping[str, str] | None = None,
) -> DatasetConfig:
    """Load and strictly validate dataset YAML configuration."""
    raw = load_yaml(path, overrides)
    _validate_keys(raw, required={"paths", "dataset"}, optional={"environment"}, location="root")

    environment_value = raw.get("environment")
    if environment_value is not None and not isinstance(environment_value, str):
        raise ConfigurationError("environment must be a string")
    environment = detect_environment(environment_value, environ=environ)

    paths_raw = _mapping(raw["paths"], "paths")
    paths = ProjectPaths.from_mapping(paths_raw, config_file=path)

    dataset = _mapping(raw["dataset"], "dataset")
    _validate_keys(
        dataset,
        required={"name", "version", "license", "created_at", "sources"},
        optional={"description"},
        location="dataset",
    )
    metadata = _metadata(dataset)
    sources = _sources(dataset["sources"], config_file=path)
    return DatasetConfig(environment, paths, metadata, sources)


def _metadata(values: ConfigMapping) -> DatasetMetadata:
    strings: dict[str, str] = {}
    for key in ("name", "version", "license"):
        value = values[key]
        if not isinstance(value, str):
            raise ConfigurationError(f"dataset.{key} must be a string")
        strings[key] = value
    description = values.get("description", "")
    if not isinstance(description, str):
        raise ConfigurationError("dataset.description must be a string")
    created_at = values["created_at"]
    if not isinstance(created_at, str):
        raise ConfigurationError("dataset.created_at must be an ISO-8601 string")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return DatasetMetadata(
            name=strings["name"],
            version=strings["version"],
            license=strings["license"],
            description=description,
            created_at=timestamp,
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid dataset metadata: {error}") from error


def _sources(value: ConfigValue, *, config_file: Path) -> tuple[DatasetSource, ...]:
    if not isinstance(value, list):
        raise ConfigurationError("dataset.sources must be a list")
    sources: list[DatasetSource] = []
    for index, item in enumerate(value):
        source = _mapping(item, f"dataset.sources[{index}]")
        _validate_keys(
            source,
            required={"path", "format", "split"},
            optional=set(),
            location=f"dataset.sources[{index}]",
        )
        raw_path = source["path"]
        raw_format = source["format"]
        raw_split = source["split"]
        if (
            not isinstance(raw_path, str)
            or not isinstance(raw_format, str)
            or not isinstance(raw_split, str)
        ):
            raise ConfigurationError(f"dataset.sources[{index}] values must be strings")
        try:
            sources.append(
                DatasetSource(
                    path=resolve_config_path(raw_path, config_file),
                    format=SourceFormat(raw_format),
                    split=DatasetSplit(raw_split),
                )
            )
        except ValueError as error:
            raise ConfigurationError(f"Invalid dataset.sources[{index}]: {error}") from error
    return tuple(sources)


def _mapping(value: ConfigValue, location: str) -> ConfigMapping:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


def _validate_keys(
    values: Mapping[str, ConfigValue],
    *,
    required: set[str],
    optional: set[str],
    location: str,
) -> None:
    keys = set(values)
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        raise ConfigurationError(
            f"{location} keys are invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
