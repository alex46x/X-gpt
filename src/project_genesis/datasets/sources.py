"""Validated local dataset source and split definitions."""

import codecs
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class SourceFormat(StrEnum):
    """Supported source declarations; parsing is introduced in later phases."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PDF = "pdf"
    GIT = "git"
    WEB = "web"


class DatasetSplit(StrEnum):
    """Canonical dataset splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """One local file or directory declared as dataset input."""

    path: Path
    format: SourceFormat
    split: DatasetSplit
    language: str = "und"
    license: str | None = None
    text_field: str = "text"
    encoding: str = "utf-8"
    include_extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the local source boundary."""
        if not self.path.is_absolute():
            raise ValueError("dataset source path must be absolute")
        if not self.path.exists():
            raise ValueError(f"dataset source does not exist: {self.path}")
        if self.path.is_symlink():
            raise ValueError(f"dataset source cannot be a symbolic link: {self.path}")
        if not (self.path.is_file() or self.path.is_dir()):
            raise ValueError(f"dataset source must be a file or directory: {self.path}")
        if not self.language.strip():
            raise ValueError("dataset source language must be non-empty")
        if self.license is not None and not self.license.strip():
            raise ValueError("dataset source license must be non-empty when provided")
        if not self.text_field.strip():
            raise ValueError("dataset source text_field must be non-empty")
        try:
            codecs.lookup(self.encoding)
        except LookupError as error:
            raise ValueError(f"unknown dataset source encoding: {self.encoding}") from error
        if any(
            not extension.startswith(".") or extension != extension.lower()
            for extension in self.include_extensions
        ):
            raise ValueError(
                "include_extensions must contain lowercase extensions beginning with '.'"
            )


def source_files(source: DatasetSource) -> tuple[Path, ...]:
    """Return the source's deterministic, validated local file selection."""
    candidates = (source.path,) if source.path.is_file() else tuple(source.path.rglob("*"))
    if symbolic_link := next((path for path in candidates if path.is_symlink()), None):
        raise ValueError(f"symbolic links are not valid dataset files: {symbolic_link}")
    files = (
        path
        for path in candidates
        if path.is_file()
        and ".git" not in path.parts
        and (not source.include_extensions or path.suffix.lower() in source.include_extensions)
    )
    return tuple(sorted(files, key=lambda path: path.as_posix()))
