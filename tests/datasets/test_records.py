from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from project_genesis.datasets import Dataset, DatasetMetadata, DatasetRecord

NOW = datetime(2026, 7, 19, tzinfo=UTC)


def make_record(document_id: str, text: str = "hello") -> DatasetRecord:
    return DatasetRecord.create(
        text=text,
        language="en",
        source="local.txt",
        license="MIT",
        document_id=document_id,
        created_at=NOW,
        metadata={"quality": 1, "tags": ("source", "test")},
    )


def make_metadata() -> DatasetMetadata:
    return DatasetMetadata("corpus", "1.0.0", "MIT", NOW)


def test_record_is_immutable_and_checksum_is_derived() -> None:
    record = make_record("doc-1")

    with pytest.raises(FrozenInstanceError):
        record.text = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.metadata["quality"] = 2  # type: ignore[index]


def test_record_rejects_checksum_mismatch_and_naive_timestamp() -> None:
    record = make_record("doc-1")

    with pytest.raises(ValueError, match="does not match"):
        DatasetRecord(
            text="changed",
            language=record.language,
            source=record.source,
            license=record.license,
            document_id=record.document_id,
            checksum=record.checksum,
            created_at=record.created_at,
        )
    with pytest.raises(ValueError, match="timezone"):
        DatasetRecord.create(
            text="hello",
            language="en",
            source="local.txt",
            license="MIT",
            document_id="doc-2",
            created_at=datetime(2026, 7, 19),
        )


def test_dataset_detects_duplicates_and_reports_statistics() -> None:
    dataset = Dataset(
        make_metadata(), (make_record("a"), make_record("b"), make_record("c", "world"))
    )
    statistics = dataset.statistics()

    assert dataset.duplicate_checksums() == (dataset[0].checksum,)
    assert statistics.records == 3
    assert statistics.characters == 15
    assert statistics.languages == {"en": 3}
    assert statistics.unique_checksums == 2

    with pytest.raises(ValueError, match="duplicate document ids"):
        Dataset(make_metadata(), (make_record("same"), make_record("same", "other")))


def test_dataset_fingerprint_depends_on_record_order() -> None:
    first = make_record("first")
    second = make_record("second", "world")

    assert (
        Dataset(make_metadata(), (first, second)).fingerprint
        != Dataset(make_metadata(), (second, first)).fingerprint
    )
