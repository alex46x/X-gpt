from pathlib import Path

import pytest

from project_genesis.configuration import ConfigurationError, RuntimeEnvironment
from project_genesis.datasets import DatasetSplit, SourceFormat, load_dataset_config


def test_dataset_configuration_is_typed_and_resolves_source_paths(tmp_path: Path) -> None:
    source = tmp_path / "documents.txt"
    source.write_text("document", encoding="utf-8")
    config_file = tmp_path / "dataset.yaml"
    config_file.write_text(
        """
environment: development
paths:
  data: data
  cache: cache
  artifacts: artifacts
dataset:
  name: corpus
  version: 1.2.3
  license: MIT
  created_at: "2026-07-19T00:00:00Z"
  sources:
    - path: documents.txt
      format: text
      split: train
      language: en
      license: Apache-2.0
      text_field: body
      encoding: utf-8
      include_extensions: [.txt]
""".lstrip(),
        encoding="utf-8",
    )

    config = load_dataset_config(
        config_file,
        ["dataset.version=2.0.0"],
        environ={},
    )

    assert config.environment is RuntimeEnvironment.DEVELOPMENT
    assert config.metadata.version == "2.0.0"
    assert config.sources[0].path == source.resolve()
    assert config.sources[0].format is SourceFormat.TEXT
    assert config.sources[0].split is DatasetSplit.TRAIN
    assert config.sources[0].language == "en"
    assert config.sources[0].license == "Apache-2.0"
    assert config.sources[0].text_field == "body"
    assert config.sources[0].include_extensions == (".txt",)


def test_dataset_configuration_rejects_unknown_fields(tmp_path: Path) -> None:
    config_file = tmp_path / "dataset.yaml"
    config_file.write_text(
        """
paths:
  data: data
  cache: cache
  artifacts: artifacts
dataset:
  name: corpus
  version: 1.0.0
  license: MIT
  created_at: "2026-07-19T00:00:00Z"
  sources: []
  misspelled: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unknown=\\['misspelled'\\]"):
        load_dataset_config(config_file, environ={})


def test_dataset_configuration_rejects_null_required_source_options(tmp_path: Path) -> None:
    source = tmp_path / "documents.txt"
    source.write_text("document", encoding="utf-8")
    config_file = tmp_path / "dataset.yaml"
    config_file.write_text(
        """
paths:
  data: data
  cache: cache
  artifacts: artifacts
dataset:
  name: corpus
  version: 1.0.0
  license: MIT
  created_at: "2026-07-19T00:00:00Z"
  sources:
    - path: documents.txt
      format: text
      split: train
      language: null
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="language cannot be null"):
        load_dataset_config(config_file, environ={})


def test_committed_default_configuration_loads() -> None:
    config = load_dataset_config(Path("configs/dataset/default.yaml"), environ={})

    assert config.metadata.name == "project-genesis"
    assert config.sources == ()
