import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from project_genesis.datasets import (
    DatasetManifest,
    DatasetMetadata,
    DatasetRegistry,
    DatasetSource,
    DatasetSplit,
    LocalManifestStorage,
    SourceFormat,
)

NOW = datetime(2026, 7, 19, tzinfo=UTC)


def build_manifest(tmp_path: Path) -> DatasetManifest:
    source = tmp_path / "source"
    source.mkdir()
    (source / "b.txt").write_text("same", encoding="utf-8")
    (source / "a.txt").write_text("same", encoding="utf-8")
    metadata = DatasetMetadata("corpus", "1.0.0", "MIT", NOW)
    return DatasetManifest.build(
        metadata=metadata,
        root=tmp_path,
        sources=(DatasetSource(source.resolve(), SourceFormat.TEXT, DatasetSplit.TRAIN),),
        created_at=NOW,
    )


def test_manifest_build_is_deterministic_and_finds_duplicate_content(tmp_path: Path) -> None:
    first = build_manifest(tmp_path)
    second = DatasetManifest.build(
        metadata=first.metadata,
        root=first.root,
        sources=(
            DatasetSource(
                (tmp_path / "source").resolve(),
                SourceFormat.TEXT,
                DatasetSplit.TRAIN,
            ),
        ),
        created_at=datetime(2030, 1, 1, tzinfo=UTC),
    )

    assert tuple(entry.path for entry in first.entries) == ("source/a.txt", "source/b.txt")
    assert first.fingerprint == second.fingerprint
    assert first.verify().duplicate_groups == (("source/a.txt", "source/b.txt"),)


def test_manifest_integrity_detects_modified_and_missing_files(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    (tmp_path / "source" / "a.txt").write_text("changed", encoding="utf-8")
    (tmp_path / "source" / "b.txt").unlink()

    report = manifest.verify()

    assert not report.is_valid
    assert report.modified == ("source/a.txt",)
    assert report.missing == ("source/b.txt",)


def test_manifest_storage_round_trips_and_rejects_tampering(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    storage = LocalManifestStorage(tmp_path / "manifests")
    path = storage.save(manifest)

    assert storage.load("corpus", "1.0.0") == manifest

    data = json.loads(path.read_text(encoding="utf-8"))
    data["entries"][0]["checksum"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        storage.load("corpus", "1.0.0")


def test_manifest_storage_rejects_unknown_fields(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    storage = LocalManifestStorage(tmp_path / "manifests")
    path = storage.save(manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["unexpected"] = True
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown=\\['unexpected'\\]"):
        storage.load("corpus", "1.0.0")


def test_registry_rejects_duplicate_dataset_versions(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    registry = DatasetRegistry()
    registry.register(manifest)

    assert registry.get("corpus", "1.0.0") is manifest
    assert registry.versions("corpus") == ("1.0.0",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(manifest)
