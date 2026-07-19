"""Validated local dataset source and split definitions."""

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
