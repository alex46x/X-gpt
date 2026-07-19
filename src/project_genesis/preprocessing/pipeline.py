"""Streaming normalization, filtering, deduplication, and quality reporting."""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from project_genesis.datasets import (
    Dataset,
    DatasetConfig,
    DatasetManifest,
    DatasetRecord,
    sha256_text,
)
from project_genesis.preprocessing.config import ErrorPolicy, PreprocessingConfig
from project_genesis.preprocessing.readers import RawDocument, ReadFailure, read_source
from project_genesis.utilities import atomic_write_text


class PreprocessingError(RuntimeError):
    """Raised when verified source data cannot be safely preprocessed."""


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Deterministic counts describing preprocessing outcomes."""

    documents_seen: int
    accepted: int
    filtered: int
    duplicates: int
    parse_failures: int
    characters_before: int
    characters_after: int
    rejection_reasons: Mapping[str, int]

    def __post_init__(self) -> None:
        """Freeze and validate report counters."""
        counters = (
            self.documents_seen,
            self.accepted,
            self.filtered,
            self.duplicates,
            self.parse_failures,
            self.characters_before,
            self.characters_after,
        )
        if any(value < 0 for value in counters):
            raise ValueError("quality report counters cannot be negative")
        object.__setattr__(
            self,
            "rejection_reasons",
            MappingProxyType(dict(sorted(self.rejection_reasons.items()))),
        )


@dataclass(frozen=True, slots=True)
class ProcessedDatasetManifest:
    """Reproducibility record linking inputs, policy, outputs, and quality."""

    dataset_name: str
    dataset_version: str
    input_fingerprint: str
    config_fingerprint: str
    output_fingerprint: str
    report: QualityReport

    @property
    def fingerprint(self) -> str:
        """Return a checksum covering the complete processed-data manifest."""
        encoded = json.dumps(self._payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible manifest data including its fingerprint."""
        return {**self._payload(), "fingerprint": self.fingerprint}

    def write(self, path: Path) -> None:
        """Atomically write the processed-data manifest as UTF-8 JSON."""
        atomic_write_text(path, json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "dataset": {"name": self.dataset_name, "version": self.dataset_version},
            "input_fingerprint": self.input_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "quality": {
                "documents_seen": self.report.documents_seen,
                "accepted": self.report.accepted,
                "filtered": self.report.filtered,
                "duplicates": self.report.duplicates,
                "parse_failures": self.report.parse_failures,
                "characters_before": self.report.characters_before,
                "characters_after": self.report.characters_after,
                "rejection_reasons": dict(self.report.rejection_reasons),
            },
        }


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Cleaned dataset with its quality and reproducibility records."""

    dataset: Dataset
    report: QualityReport
    manifest: ProcessedDatasetManifest


def preprocess_dataset(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    input_manifest: DatasetManifest,
) -> PreprocessingResult:
    """Read verified local sources into a deterministic cleaned dataset."""
    if input_manifest.metadata != dataset_config.metadata:
        raise PreprocessingError("input manifest metadata does not match dataset configuration")
    integrity = input_manifest.verify()
    if not integrity.is_valid:
        raise PreprocessingError(
            f"input manifest failed integrity verification: "
            f"missing={integrity.missing}, modified={integrity.modified}"
        )

    records: list[DatasetRecord] = []
    reasons: Counter[str] = Counter()
    seen = filtered = duplicates = failures = before = after = 0
    # ponytail: in-memory dedup; use partitioned disk-backed checksums when corpora exceed RAM.
    checksums: set[str] = set()

    for source in dataset_config.sources:
        for result in read_source(source, root=input_manifest.root):
            if isinstance(result, ReadFailure):
                failures += 1
                reasons["parse_failure"] += 1
                if preprocessing_config.on_error is ErrorPolicy.RAISE:
                    raise PreprocessingError(f"unable to parse {result.source}: {result.error}")
                continue

            seen += 1
            before += len(result.text)
            text = normalize_text(result.text, preprocessing_config)
            rejection = _rejection_reason(result, text, preprocessing_config)
            if rejection is not None:
                filtered += 1
                reasons[rejection] += 1
                continue

            checksum = sha256_text(text)
            if preprocessing_config.deduplicate and checksum in checksums:
                duplicates += 1
                reasons["duplicate"] += 1
                continue
            checksums.add(checksum)
            after += len(text)
            metadata = dict(result.metadata)
            metadata["raw_checksum"] = sha256_text(result.text)
            records.append(
                DatasetRecord(
                    text=text,
                    language=result.language,
                    source=result.source,
                    license=result.license or dataset_config.metadata.license,
                    document_id=result.document_id,
                    checksum=checksum,
                    created_at=dataset_config.metadata.created_at,
                    metadata=metadata,
                )
            )

    dataset = Dataset(dataset_config.metadata, tuple(records))
    report = QualityReport(
        documents_seen=seen,
        accepted=len(records),
        filtered=filtered,
        duplicates=duplicates,
        parse_failures=failures,
        characters_before=before,
        characters_after=after,
        rejection_reasons=dict(reasons),
    )
    processed_manifest = ProcessedDatasetManifest(
        dataset_name=dataset.metadata.name,
        dataset_version=dataset.metadata.version,
        input_fingerprint=input_manifest.fingerprint,
        config_fingerprint=_config_fingerprint(
            dataset_config,
            preprocessing_config,
            input_manifest.root,
        ),
        output_fingerprint=dataset.fingerprint,
        report=report,
    )
    return PreprocessingResult(dataset, report, processed_manifest)


def _config_fingerprint(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    root: Path,
) -> str:
    sources = [
        {
            "path": source.path.resolve().relative_to(root).as_posix(),
            "format": source.format.value,
            "split": source.split.value,
            "language": source.language,
            "license": source.license,
            "text_field": source.text_field,
            "encoding": source.encoding,
            "include_extensions": source.include_extensions,
        }
        for source in dataset_config.sources
    ]
    payload = {"preprocessing": preprocessing_config.fingerprint, "sources": sources}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(text: str, config: PreprocessingConfig) -> str:
    """Apply configured Unicode, newline, control, and whitespace normalization."""
    if config.normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize(config.unicode_normalization, text)
    if config.strip_control_characters:
        text = "".join(
            character
            for character in text
            if character in {"\n", "\t"} or not unicodedata.category(character).startswith("C")
        )
    if config.collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    return text.strip() if config.trim else text


def _rejection_reason(
    document: RawDocument,
    text: str,
    config: PreprocessingConfig,
) -> str | None:
    if len(text) < config.min_characters:
        return "too_short"
    if len(text) > config.max_characters:
        return "too_long"
    if config.allowed_languages and document.language not in config.allowed_languages:
        return "language"
    return None
