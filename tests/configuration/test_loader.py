from pathlib import Path

import pytest

from project_genesis.configuration import ConfigurationError, load_yaml


def test_load_yaml_applies_existing_dotted_overrides(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("dataset:\n  version: 1.0.0\n  enabled: false\n", encoding="utf-8")

    loaded = load_yaml(config, ["dataset.version=2.1.0", "dataset.enabled=true"])

    assert loaded == {"dataset": {"version": "2.1.0", "enabled": True}}


@pytest.mark.parametrize(
    "override",
    ["dataset.missing=value", "missing.value=1", "dataset.version", "=value"],
)
def test_load_yaml_rejects_invalid_overrides(tmp_path: Path, override: str) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("dataset:\n  version: 1.0.0\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_yaml(config, [override])


def test_load_yaml_rejects_unsafe_or_non_mapping_yaml(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text("!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")
    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_yaml(unsafe)
    with pytest.raises(ConfigurationError, match="root must be a mapping"):
        load_yaml(sequence)
