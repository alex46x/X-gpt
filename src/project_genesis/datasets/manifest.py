"""Deterministic dataset manifests and local integrity verification."""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from project_genesis.datasets.integrity import is_sha256, sha256_file
from project_genesis.datasets.records import DatasetMetadata
from project_genesis.datasets.sources import DatasetSource, DatasetSplit, SourceFormat


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """Integrity metadata for one local source file."""

    path: str
    size_bytes: int
    checksum: str
    format: SourceFormat
    split: DatasetSplit

    def __post_init__(self) -> None:
        """Validate normalized path and integrity fields."""
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or self.path != path.as_posix():
            raise ValueError(f"manifest path must be a normalized relative POSIX path: {self.path}")
        if self.size_bytes < 0:
            raise ValueError("size_bytes cannot be negative")
        if not is_sha256(self.checksum):
            raise ValueError("checksum must be a lowercase SHA-256 value")


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Result of verifying manifest entries against local storage."""

    missing: tuple[str, ...]
    modified: tuple[str, ...]
    duplicate_groups: tuple[tuple[str, ...], ...]

    @property
    def is_valid(self) -> bool:
        """Return whether every declared file exists with its expected content."""
        return not self.missing and not self.modified


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable inventory of files belonging to one dataset version."""

    metadata: DatasetMetadata
    root: Path
    created_at: datetime
    entries: tuple[ManifestEntry, ...]

    def __post_init__(self) -> None:
        """Validate manifest path, timestamp, and entry ordering."""
        if not self.root.is_absolute():
            raise ValueError("manifest root must be absolute")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("manifest created_at must include a timezone")
        paths = tuple(entry.path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("manifest entries must have unique, sorted paths")

    @classmethod
    def build(
        cls,
        *,
        metadata: DatasetMetadata,
        root: Path,
        sources: tuple[DatasetSource, ...],
        created_at: datetime,
    ) -> "DatasetManifest":
        """Inventory local source files in deterministic path order."""
        resolved_root = root.expanduser().resolve()
        if not resolved_root.is_dir():
            raise ValueError(f"manifest root must be an existing directory: {root}")

        entries: dict[str, ManifestEntry] = {}
        for source in sources:
            candidates = (source.path,) if source.path.is_file() else tuple(source.path.rglob("*"))
            if symbolic_link := next((path for path in candidates if path.is_symlink()), None):
                raise ValueError(f"symbolic links are not valid dataset files: {symbolic_link}")
            for file in sorted(
                (path for path in candidates if path.is_file()), key=lambda path: path.as_posix()
            ):
                resolved_file = file.resolve()
                try:
                    relative = resolved_file.relative_to(resolved_root).as_posix()
                except ValueError as error:
                    raise ValueError(f"dataset file is outside manifest root: {file}") from error
                if relative in entries:
                    raise ValueError(f"dataset file is declared by multiple sources: {relative}")
                entries[relative] = ManifestEntry(
                    path=relative,
                    size_bytes=resolved_file.stat().st_size,
                    checksum=sha256_file(resolved_file),
                    format=source.format,
                    split=source.split,
                )
        return cls(
            metadata, resolved_root, created_at, tuple(entries[key] for key in sorted(entries))
        )

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint excluding machine paths and timestamps."""
        payload = {
            "dataset": self.metadata.name,
            "version": self.metadata.version,
            "license": self.metadata.license,
            "description": self.metadata.description,
            "attributes": dict(self.metadata.attributes),
            "entries": [
                {
                    "path": entry.path,
                    "size_bytes": entry.size_bytes,
                    "checksum": entry.checksum,
                    "format": entry.format.value,
                    "split": entry.split.value,
                }
                for entry in self.entries
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def verify(self) -> IntegrityReport:
        """Verify presence, size, checksums, and duplicate content."""
        missing: list[str] = []
        modified: list[str] = []
        by_checksum: defaultdict[str, list[str]] = defaultdict(list)
        for entry in self.entries:
            path = self.root / PurePosixPath(entry.path)
            if not path.is_file() or path.is_symlink():
                missing.append(entry.path)
                continue
            if path.stat().st_size != entry.size_bytes or sha256_file(path) != entry.checksum:
                modified.append(entry.path)
            by_checksum[entry.checksum].append(entry.path)
        duplicates = tuple(
            tuple(paths) for _, paths in sorted(by_checksum.items()) if len(paths) > 1
        )
        return IntegrityReport(tuple(missing), tuple(modified), duplicates)

    def to_dict(self) -> dict[str, object]:
        """Convert the manifest to JSON-compatible data."""
        return {
            "dataset": {
                "name": self.metadata.name,
                "version": self.metadata.version,
                "license": self.metadata.license,
                "created_at": self.metadata.created_at.isoformat(),
                "description": self.metadata.description,
                "attributes": dict(self.metadata.attributes),
            },
            "root": str(self.root),
            "created_at": self.created_at.isoformat(),
            "fingerprint": self.fingerprint,
            "entries": [
                {
                    "path": entry.path,
                    "size_bytes": entry.size_bytes,
                    "checksum": entry.checksum,
                    "format": entry.format.value,
                    "split": entry.split.value,
                }
                for entry in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, data: object) -> "DatasetManifest":
        """Validate and construct a manifest from decoded JSON data."""
        if not isinstance(data, dict):
            raise ValueError("manifest must be a mapping")
        _require_exact_keys(
            data,
            {"dataset", "root", "created_at", "fingerprint", "entries"},
            "manifest",
        )
        dataset = _dict_field(data, "dataset")
        _require_exact_keys(
            dataset,
            {"name", "version", "license", "created_at", "description", "attributes"},
            "dataset",
        )
        entries_data = _list_field(data, "entries")
        metadata = DatasetMetadata(
            name=_string_field(dataset, "name"),
            version=_string_field(dataset, "version"),
            license=_string_field(dataset, "license"),
            created_at=_timestamp_field(dataset, "created_at"),
            description=_string_field(dataset, "description"),
            attributes=_attributes_field(dataset, "attributes"),
        )
        entries_list: list[ManifestEntry] = []
        for item in entries_data:
            entry = _require_dict(item, "manifest entry")
            _require_exact_keys(
                entry,
                {"path", "size_bytes", "checksum", "format", "split"},
                "manifest entry",
            )
            entries_list.append(
                ManifestEntry(
                    path=_string_field(entry, "path"),
                    size_bytes=_integer_field(entry, "size_bytes"),
                    checksum=_string_field(entry, "checksum"),
                    format=SourceFormat(_string_field(entry, "format")),
                    split=DatasetSplit(_string_field(entry, "split")),
                )
            )
        manifest = cls(
            metadata=metadata,
            root=Path(_string_field(data, "root")).expanduser().resolve(),
            created_at=_timestamp_field(data, "created_at"),
            entries=tuple(entries_list),
        )
        if _string_field(data, "fingerprint") != manifest.fingerprint:
            raise ValueError("manifest fingerprint does not match its contents")
        return manifest


def _require_dict(value: object, location: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return value


def _require_exact_keys(
    data: dict[object, object],
    expected: set[str],
    location: str,
) -> None:
    keys = set(data)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(str(key) for key in keys - expected)
        raise ValueError(f"{location} keys are invalid; missing={missing}, unknown={unknown}")


def _dict_field(data: dict[object, object], key: str) -> dict[object, object]:
    return _require_dict(data.get(key), key)


def _list_field(data: dict[object, object], key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _string_field(data: dict[object, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _integer_field(data: dict[object, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _timestamp_field(data: dict[object, object], key: str) -> datetime:
    try:
        return datetime.fromisoformat(_string_field(data, key).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{key} must be an ISO-8601 timestamp") from error


def _attributes_field(
    data: dict[object, object],
    key: str,
) -> dict[str, str | int | float | bool | None | tuple[str | int | float | bool | None, ...]]:
    values = _dict_field(data, key)
    attributes: dict[
        str, str | int | float | bool | None | tuple[str | int | float | bool | None, ...]
    ] = {}
    for name, value in values.items():
        if not isinstance(name, str):
            raise ValueError("attribute names must be strings")
        if isinstance(value, list):
            if not all(
                item is None or isinstance(item, str | int | float | bool) for item in value
            ):
                raise ValueError(f"attribute {name!r} contains unsupported values")
            attributes[name] = tuple(value)
        elif value is None or isinstance(value, str | int | float | bool):
            attributes[name] = value
        else:
            raise ValueError(f"attribute {name!r} has an unsupported value")
    return attributes
