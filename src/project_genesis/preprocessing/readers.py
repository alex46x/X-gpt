"""Deterministic local readers for declared dataset source formats."""

import csv
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from types import MappingProxyType

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from project_genesis.datasets import DatasetSource, MetadataValue, SourceFormat
from project_genesis.datasets.sources import source_files


@dataclass(frozen=True, slots=True)
class RawDocument:
    """One parsed but unnormalized local document."""

    text: str
    document_id: str
    source: str
    language: str
    license: str | None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze parser metadata."""
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ReadFailure:
    """A source file that could not be decoded or parsed."""

    source: str
    error: str


type ReadResult = RawDocument | ReadFailure


def read_source(source: DatasetSource, *, root: Path) -> Iterator[ReadResult]:
    """Yield parsed documents or file-level failures in deterministic order."""
    resolved_root = root.expanduser().resolve()
    for path in source_files(source):
        try:
            relative = path.resolve().relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise ValueError(f"source file is outside dataset root: {path}") from error
        try:
            yield from _read_file(path, relative, source)
        except (
            OSError,
            UnicodeError,
            csv.Error,
            json.JSONDecodeError,
            PdfReadError,
            ValueError,
        ) as error:
            yield ReadFailure(relative, f"{type(error).__name__}: {error}")


def _read_file(path: Path, relative: str, source: DatasetSource) -> Iterator[RawDocument]:
    if source.format in {SourceFormat.TEXT, SourceFormat.MARKDOWN, SourceFormat.GIT}:
        yield _document(path.read_text(encoding=source.encoding), relative, "0", source)
    elif source.format is SourceFormat.WEB:
        parser = _VisibleTextParser()
        parser.feed(path.read_text(encoding=source.encoding))
        parser.close()
        yield _document(parser.text, relative, "0", source)
    elif source.format is SourceFormat.JSON:
        with path.open(encoding=source.encoding) as stream:
            value: object = json.load(stream, parse_constant=_reject_json_constant)
        for ordinal, text, metadata in _json_documents(value, source.text_field):
            yield _document(text, relative, ordinal, source, metadata)
    elif source.format is SourceFormat.JSONL:
        with path.open(encoding=source.encoding) as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line, parse_constant=_reject_json_constant)
                documents = tuple(_json_documents(value, source.text_field))
                if len(documents) != 1:
                    raise ValueError(f"JSONL line {line_number} must contain one document")
                _, text, metadata = documents[0]
                yield _document(text, relative, f"line-{line_number}", source, metadata)
    elif source.format is SourceFormat.CSV:
        with path.open(encoding=source.encoding, newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or source.text_field not in reader.fieldnames:
                raise ValueError(f"CSV must contain text field {source.text_field!r}")
            for row_number, row in enumerate(reader, start=2):
                text = row.pop(source.text_field)
                if not isinstance(text, str):
                    raise ValueError(f"CSV row {row_number} has no text")
                yield _document(
                    text,
                    relative,
                    f"row-{row_number}",
                    source,
                    _metadata(row),
                )
    elif source.format is SourceFormat.PDF:
        for page_number, page in enumerate(PdfReader(path).pages, start=1):
            yield _document(
                page.extract_text() or "",
                relative,
                f"page-{page_number}",
                source,
                {"page": page_number},
            )
    else:
        raise ValueError(f"unsupported source format: {source.format}")


def _json_documents(
    value: object,
    text_field: str,
) -> Iterator[tuple[str, str, Mapping[str, MetadataValue]]]:
    items = value if isinstance(value, list) else [value]
    for index, item in enumerate(items):
        if isinstance(item, str):
            yield str(index), item, {}
            continue
        if not isinstance(item, dict):
            raise ValueError("JSON documents must be strings or objects")
        fields = dict(item)
        text = fields.pop(text_field, None)
        if not isinstance(text, str):
            raise ValueError(f"JSON document must contain string field {text_field!r}")
        yield str(index), text, _metadata(fields)


def _metadata(values: Mapping[object, object]) -> Mapping[str, MetadataValue]:
    metadata: dict[str, MetadataValue] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError("document metadata keys must be strings")
        if isinstance(value, list):
            if not all(
                item is None or isinstance(item, str | int | float | bool) for item in value
            ):
                raise ValueError(f"document metadata {key!r} contains nested values")
            metadata[key] = tuple(value)
        elif value is None or isinstance(value, str | int | float | bool):
            metadata[key] = value
        else:
            raise ValueError(f"document metadata {key!r} contains nested values")
    return metadata


def _document(
    text: str,
    relative: str,
    ordinal: str,
    source: DatasetSource,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> RawDocument:
    values = {} if metadata is None else dict(metadata)
    values.update({"format": source.format.value, "split": source.split.value})
    return RawDocument(
        text=text,
        document_id=f"{relative}#{ordinal}",
        source=relative,
        language=source.language,
        license=source.license,
        metadata=values,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not supported: {value}")


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._parts)
