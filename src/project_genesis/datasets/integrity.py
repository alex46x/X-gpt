"""Checksum primitives for dataset records and files."""

import hashlib
import re
from pathlib import Path

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_text(text: str) -> str:
    """Return the SHA-256 checksum of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's SHA-256 checksum without loading it entirely in memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: str) -> bool:
    """Return whether a value is a lowercase SHA-256 checksum."""
    return SHA256_PATTERN.fullmatch(value) is not None
