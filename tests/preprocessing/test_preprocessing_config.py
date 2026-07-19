from pathlib import Path

import pytest

from project_genesis.configuration import ConfigurationError
from project_genesis.preprocessing import (
    ErrorPolicy,
    PreprocessingConfig,
    load_preprocessing_config,
)


def test_default_preprocessing_configuration_loads_and_overrides() -> None:
    config = load_preprocessing_config(
        Path("configs/preprocessing/default.yaml"),
        ["preprocessing.min_characters=20", "preprocessing.on_error=skip"],
    )

    assert config.min_characters == 20
    assert config.on_error is ErrorPolicy.SKIP
    assert len(config.fingerprint) == 64


def test_preprocessing_configuration_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="max_characters"):
        PreprocessingConfig(
            unicode_normalization="NFKC",
            normalize_newlines=True,
            strip_control_characters=True,
            collapse_whitespace=False,
            trim=True,
            min_characters=10,
            max_characters=5,
            allowed_languages=(),
            deduplicate=True,
            on_error=ErrorPolicy.RAISE,
        )


def test_preprocessing_configuration_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "preprocessing.yaml"
    config.write_text(
        """
preprocessing:
  unicode_normalization: NFKC
  normalize_newlines: true
  strip_control_characters: true
  collapse_whitespace: false
  trim: true
  min_characters: 1
  max_characters: 10
  allowed_languages: []
  deduplicate: true
  on_error: raise
  unknown: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unknown=\\['unknown'\\]"):
        load_preprocessing_config(config)
