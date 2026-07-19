"""Typed dataset configuration decoded from safe YAML."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import overload

from project_genesis.configuration import (
    ConfigMapping,
    ConfigurationError,
    ConfigValue,
    ProjectPaths,
    RuntimeEnvironment,
    detect_environment,
    load_yaml,
    require_mapping,
    resolve_config_path,
    validate_keys,
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
    validate_keys(raw, required={"paths", "dataset"}, optional={"environment"}, location="root")

    environment_value = raw.get("environment")
    if environment_value is not None and not isinstance(environment_value, str):
        raise ConfigurationError("environment must be a string")
    environment = detect_environment(environment_value, environ=environ)

    paths_raw = require_mapping(raw["paths"], "paths")
    paths = ProjectPaths.from_mapping(paths_raw, config_file=path)

    dataset = require_mapping(raw["dataset"], "dataset")
    validate_keys(
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
        source = require_mapping(item, f"dataset.sources[{index}]")
        validate_keys(
            source,
            required={"path", "format", "split"},
            optional={"language", "license", "text_field", "encoding", "include_extensions"},
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
            language = _optional_string(source, "language", "und")
            license_name = _optional_string(source, "license", None)
            text_field = _optional_string(source, "text_field", "text")
            encoding = _optional_string(source, "encoding", "utf-8")
            extensions = _string_tuple(source.get("include_extensions", []), "include_extensions")
            sources.append(
                DatasetSource(
                    path=resolve_config_path(raw_path, config_file),
                    format=SourceFormat(raw_format),
                    split=DatasetSplit(raw_split),
                    language=language,
                    license=license_name,
                    text_field=text_field,
                    encoding=encoding,
                    include_extensions=extensions,
                )
            )
        except ValueError as error:
            raise ConfigurationError(f"Invalid dataset.sources[{index}]: {error}") from error
    return tuple(sources)


@overload
def _optional_string(
    values: ConfigMapping,
    key: str,
    default: str,
) -> str: ...


@overload
def _optional_string(
    values: ConfigMapping,
    key: str,
    default: None,
) -> str | None: ...


def _optional_string(
    values: ConfigMapping,
    key: str,
    default: str | None,
) -> str | None:
    value = values.get(key, default)
    if value is None and default is not None:
        raise ConfigurationError(f"dataset source {key} cannot be null")
    if value is not None and not isinstance(value, str):
        raise ConfigurationError(f"dataset source {key} must be a string")
    return value


def _string_tuple(value: ConfigValue, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"dataset source {location} must be a list of strings")
    return tuple(item for item in value if isinstance(item, str))
