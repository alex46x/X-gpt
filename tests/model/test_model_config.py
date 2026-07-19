from pathlib import Path

import pytest

from project_genesis.configuration import ConfigurationError
from project_genesis.model import ModelConfig, load_model_config


def test_default_model_configuration_loads_with_override() -> None:
    config = load_model_config(
        Path("configs/model/default.yaml"),
        ["model.context_length=1024", "model.dropout=0.0"],
    )

    assert config.context_length == 1024
    assert config.dropout == 0.0
    assert config.head_dim == 64


def test_model_configuration_rejects_incompatible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(100, 16, 10, 3, 40, 0.0, True, 1e-5)


def test_model_configuration_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
model:
  vocab_size: 100
  context_length: 16
  d_model: 8
  n_heads: 2
  d_ff: 32
  dropout: 0.0
  bias: true
  layer_norm_epsilon: 0.00001
  unknown: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unknown=\\['unknown'\\]"):
        load_model_config(path)
