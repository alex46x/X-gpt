import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from project_genesis.configuration import ProjectPaths, RuntimeEnvironment
from project_genesis.datasets import (
    DatasetConfig,
    DatasetManifest,
    DatasetMetadata,
    DatasetSource,
    DatasetSplit,
    SourceFormat,
)
from project_genesis.preprocessing import (
    ErrorPolicy,
    PreprocessingConfig,
    PreprocessingError,
    preprocess_dataset,
)

NOW = datetime(2026, 7, 19, tzinfo=UTC)


def policy(**changes: object) -> PreprocessingConfig:
    values: dict[str, object] = {
        "unicode_normalization": "NFKC",
        "normalize_newlines": True,
        "strip_control_characters": True,
        "collapse_whitespace": False,
        "trim": True,
        "min_characters": 3,
        "max_characters": 100,
        "allowed_languages": ("en",),
        "deduplicate": True,
        "on_error": ErrorPolicy.RAISE,
    }
    values.update(changes)
    return PreprocessingConfig(**values)  # type: ignore[arg-type]


def setup_dataset(
    tmp_path: Path, content: str, *, format: SourceFormat = SourceFormat.JSONL
) -> tuple[DatasetConfig, DatasetManifest]:
    source_path = tmp_path / f"source.{format.value}"
    source_path.write_text(content, encoding="utf-8")
    metadata = DatasetMetadata("corpus", "1.0.0", "MIT", NOW)
    source = DatasetSource(
        source_path.resolve(),
        format,
        DatasetSplit.TRAIN,
        language="en",
    )
    config = DatasetConfig(
        RuntimeEnvironment.TEST,
        ProjectPaths(tmp_path, tmp_path / "cache", tmp_path / "artifacts"),
        metadata,
        (source,),
    )
    manifest = DatasetManifest.build(
        metadata=metadata,
        root=tmp_path,
        sources=(source,),
        created_at=NOW,
    )
    return config, manifest


def test_pipeline_normalizes_filters_deduplicates_and_writes_manifest(tmp_path: Path) -> None:
    config, input_manifest = setup_dataset(
        tmp_path,
        '{"text": "  Héllo\\r\\n"}\n{"text": "Héllo"}\n{"text": "x"}\n',
    )

    result = preprocess_dataset(config, policy(), input_manifest)
    output = tmp_path / "processed.json"
    result.manifest.write(output)

    assert [record.text for record in result.dataset] == ["Héllo"]
    assert result.dataset[0].token_ids is None
    assert result.report.accepted == 1
    assert result.report.filtered == 1
    assert result.report.duplicates == 1
    assert result.report.rejection_reasons == {"duplicate": 1, "too_short": 1}
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["output_fingerprint"] == result.dataset.fingerprint
    assert persisted["fingerprint"] == result.manifest.fingerprint


def test_pipeline_result_is_deterministic(tmp_path: Path) -> None:
    config, input_manifest = setup_dataset(tmp_path, '{"text": "one"}\n{"text": "two"}\n')

    first = preprocess_dataset(config, policy(), input_manifest)
    second = preprocess_dataset(config, policy(), input_manifest)

    assert first.dataset.fingerprint == second.dataset.fingerprint
    assert first.manifest.fingerprint == second.manifest.fingerprint


def test_pipeline_requires_valid_input_manifest(tmp_path: Path) -> None:
    config, input_manifest = setup_dataset(tmp_path, '{"text": "valid"}\n')
    config.sources[0].path.write_text('{"text": "changed"}\n', encoding="utf-8")

    with pytest.raises(PreprocessingError, match="integrity verification"):
        preprocess_dataset(config, policy(), input_manifest)


def test_pipeline_can_report_and_skip_parser_failure(tmp_path: Path) -> None:
    config, input_manifest = setup_dataset(tmp_path, "{bad", format=SourceFormat.JSON)

    result = preprocess_dataset(
        config,
        policy(on_error=ErrorPolicy.SKIP),
        input_manifest,
    )

    assert result.report.parse_failures == 1
    assert result.report.rejection_reasons == {"parse_failure": 1}
