from pathlib import Path

import pytest

from project_genesis.configuration import ConfigurationError
from project_genesis.tokenizer import (
    SpecialTokens,
    TokenizerConfig,
    load_tokenizer_config,
)


def test_default_tokenizer_configuration_loads_with_override() -> None:
    config = load_tokenizer_config(
        Path("configs/tokenizer/default.yaml"),
        ["tokenizer.vocab_size=1024", "tokenizer.add_eos=true"],
    )

    assert config.vocab_size == 1024
    assert config.add_eos
    assert config.special_tokens.values == ("<pad>", "<bos>", "<eos>", "<unk>")
    assert len(config.fingerprint) == 64


def test_tokenizer_configuration_rejects_small_vocabulary() -> None:
    with pytest.raises(ValueError, match="at least 260"):
        TokenizerConfig(
            vocab_size=259,
            min_pair_frequency=2,
            special_tokens=SpecialTokens("<pad>", "<bos>", "<eos>", "<unk>"),
            add_bos=False,
            add_eos=False,
        )


def test_tokenizer_configuration_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "tokenizer.yaml"
    config.write_text(
        """
tokenizer:
  vocab_size: 300
  min_pair_frequency: 2
  special_tokens:
    pad: "<pad>"
    bos: "<bos>"
    eos: "<eos>"
    unk: "<unk>"
  add_bos: false
  add_eos: false
  unknown: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unknown=\\['unknown'\\]"):
        load_tokenizer_config(config)
