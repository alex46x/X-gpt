"""Immutable dataset records, metadata, schema, and collection contracts."""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import overload

from project_genesis.datasets.integrity import is_sha256, sha256_text

type MetadataScalar = str | int | float | bool | None
type MetadataValue = MetadataScalar | tuple[MetadataScalar, ...]
type DatasetCache = MutableMapping[str, Dataset]

SEMANTIC_VERSION = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One immutable source document and its provenance."""

    text: str
    language: str
    source: str
    license: str
    document_id: str
    checksum: str
    created_at: datetime
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    token_ids: tuple[int, ...] | None = None
    embeddings: tuple[tuple[float, ...], ...] | None = None
    labels: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate record content and freeze metadata."""
        for name in ("text", "language", "source", "license", "document_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not is_sha256(self.checksum):
            raise ValueError("checksum must be a lowercase SHA-256 value")
        if self.checksum != sha256_text(self.text):
            raise ValueError("checksum does not match record text")
        _require_aware_timestamp(self.created_at)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        if self.token_ids is not None and any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token_ids must contain non-negative integers")
        if self.labels is not None and any(label < 0 for label in self.labels):
            raise ValueError("labels must contain non-negative integers")
        if self.embeddings is not None:
            widths = {len(row) for row in self.embeddings}
            if len(widths) > 1 or any(
                not math.isfinite(value) for row in self.embeddings for value in row
            ):
                raise ValueError("embeddings must be rectangular and contain finite values")

    @classmethod
    def create(
        cls,
        *,
        text: str,
        language: str,
        source: str,
        license: str,
        document_id: str,
        created_at: datetime,
        metadata: Mapping[str, MetadataValue] | None = None,
    ) -> "DatasetRecord":
        """Create a source record with a checksum derived from its exact text."""
        return cls(
            text=text,
            language=language,
            source=source,
            license=license,
            document_id=document_id,
            checksum=sha256_text(text),
            created_at=created_at,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Identity and provenance shared by one versioned dataset."""

    name: str
    version: str
    license: str
    created_at: datetime
    description: str = ""
    attributes: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dataset identity and freeze attributes."""
        for name in ("name", "version", "license"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if SEMANTIC_VERSION.fullmatch(self.version) is None:
            raise ValueError("version must follow semantic versioning")
        _require_aware_timestamp(self.created_at)
        object.__setattr__(self, "attributes", _freeze_metadata(self.attributes))


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Versioned field contract for dataset records."""

    version: str = "1.0.0"
    required_fields: tuple[str, ...] = (
        "text",
        "language",
        "source",
        "license",
        "document_id",
        "checksum",
        "created_at",
        "metadata",
    )
    optional_fields: tuple[str, ...] = ("token_ids", "embeddings", "labels")

    def __post_init__(self) -> None:
        """Validate schema version and field separation."""
        if SEMANTIC_VERSION.fullmatch(self.version) is None:
            raise ValueError("schema version must follow semantic versioning")
        if not self.required_fields or set(self.required_fields) & set(self.optional_fields):
            raise ValueError("schema fields must be non-empty and disjoint")


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """Cheap statistics available before cleaning or tokenization."""

    records: int
    characters: int
    languages: Mapping[str, int]
    unique_checksums: int

    def __post_init__(self) -> None:
        """Freeze the language count mapping."""
        object.__setattr__(self, "languages", MappingProxyType(dict(self.languages)))


@dataclass(frozen=True, slots=True)
class Dataset:
    """An immutable, validated collection of source records."""

    metadata: DatasetMetadata
    records: tuple[DatasetRecord, ...]
    schema: DatasetSchema = field(default_factory=DatasetSchema)

    def __post_init__(self) -> None:
        """Reject ambiguous document identities."""
        duplicate_ids = self.duplicate_document_ids()
        if duplicate_ids:
            raise ValueError(f"duplicate document ids: {', '.join(duplicate_ids)}")

    def __len__(self) -> int:
        """Return the record count."""
        return len(self.records)

    def __iter__(self) -> Iterator[DatasetRecord]:
        """Iterate records in stable order."""
        return iter(self.records)

    @overload
    def __getitem__(self, index: int) -> DatasetRecord: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[DatasetRecord, ...]: ...

    def __getitem__(self, index: int | slice) -> DatasetRecord | tuple[DatasetRecord, ...]:
        """Return a record or immutable record slice."""
        return self.records[index]

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint of schema, metadata, and record order."""
        payload = {
            "dataset": self.metadata.name,
            "version": self.metadata.version,
            "schema": self.schema.version,
            "records": [
                {"document_id": record.document_id, "checksum": record.checksum}
                for record in self.records
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def statistics(self) -> DatasetStatistics:
        """Calculate deterministic pre-tokenization statistics."""
        languages = Counter(record.language for record in self.records)
        return DatasetStatistics(
            records=len(self.records),
            characters=sum(len(record.text) for record in self.records),
            languages=dict(sorted(languages.items())),
            unique_checksums=len({record.checksum for record in self.records}),
        )

    def duplicate_document_ids(self) -> tuple[str, ...]:
        """Return document identifiers occurring more than once."""
        counts = Counter(record.document_id for record in self.records)
        return tuple(sorted(key for key, count in counts.items() if count > 1))

    def duplicate_checksums(self) -> tuple[str, ...]:
        """Return content checksums occurring more than once."""
        counts = Counter(record.checksum for record in self.records)
        return tuple(sorted(key for key, count in counts.items() if count > 1))


def _require_aware_timestamp(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at must include a timezone")


def _freeze_metadata(values: Mapping[str, MetadataValue]) -> Mapping[str, MetadataValue]:
    frozen: dict[str, MetadataValue] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if isinstance(value, tuple):
            if not all(
                item is None or isinstance(item, str | int | float | bool) for item in value
            ):
                raise ValueError(f"metadata value for {key!r} contains an unsupported item")
        elif value is not None and not isinstance(value, str | int | float | bool):
            raise ValueError(f"metadata value for {key!r} has an unsupported type")
        frozen[key] = value
    return MappingProxyType(frozen)
