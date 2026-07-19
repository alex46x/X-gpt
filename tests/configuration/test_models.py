from pathlib import Path

import pytest

from project_genesis.configuration import (
    ConfigurationError,
    ProjectPaths,
    RuntimeEnvironment,
    detect_environment,
)


def test_environment_variable_precedes_configured_environment() -> None:
    environment = detect_environment(
        "production",
        environ={"PROJECT_GENESIS_ENV": "test"},
    )

    assert environment is RuntimeEnvironment.TEST


def test_environment_rejects_unknown_value() -> None:
    with pytest.raises(ConfigurationError, match="must be one of"):
        detect_environment(environ={"PROJECT_GENESIS_ENV": "staging"})


def test_project_paths_resolve_from_declaring_file(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "dataset.yaml"
    paths = ProjectPaths.from_mapping(
        {"data": "../data", "cache": "../cache", "artifacts": "../artifacts"},
        config_file=config,
    )

    assert paths.data == (tmp_path / "data").resolve()
    assert paths.cache == (tmp_path / "cache").resolve()
    assert paths.artifacts == (tmp_path / "artifacts").resolve()
