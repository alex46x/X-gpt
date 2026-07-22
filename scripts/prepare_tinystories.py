"""Materialize a pinned 10-million-word TinyStories English subset."""

import argparse
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "roneneldan/TinyStories"
SOURCE_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SOURCE_SIZE = 2_227_753_162
SOURCE_XET_HASH = "02e40cc51c59a4bc6c51bd7bc9acda4316e208745be060558eaf500cd14e9f96"
SOURCE_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    f"{SOURCE_REVISION}/TinyStoriesV2-GPT4-train.txt"
)
VALIDATION_SIZE = 22_502_601
VALIDATION_XET_HASH = "e9c9ab082c52b89a2e85b03407638201d088148e94dccd9b127c60226e2a51bf"
VALIDATION_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    f"{SOURCE_REVISION}/TinyStoriesV2-GPT4-valid.txt"
)
RANGE_BYTES = 128 * 1024 * 1024


def main() -> None:
    """Download deterministic prefixes and write train and validation JSONL files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/tinystories-10m")
    parser.add_argument("--training-words", type=int, default=10_000_000)
    parser.add_argument("--validation-stories", type=int, default=1_000)
    args = parser.parse_args()
    if args.training_words < 1 or args.validation_stories < 1:
        parser.error("training-words and validation-stories must be positive")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = _materialize(
        output / "train.jsonl",
        SOURCE_URL,
        SOURCE_SIZE,
        SOURCE_XET_HASH,
        word_limit=args.training_words,
    )
    validation = _materialize(
        output / "validation.jsonl",
        VALIDATION_URL,
        VALIDATION_SIZE,
        VALIDATION_XET_HASH,
        story_limit=args.validation_stories,
    )
    manifest = {
        "dataset": DATASET,
        "source_revision": SOURCE_REVISION,
        "license": "CDLA-Sharing-1.0",
        "selection": "deterministic source prefix of complete stories",
        "splits": {
            "train": {**train, "sha256": _sha256(output / "train.jsonl")},
            "validation": {
                **validation,
                "sha256": _sha256(output / "validation.jsonl"),
            },
        },
    }
    _write_text(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"train: {train['stories']:,} stories, {train['words']:,} words")
    print(f"validation: {validation['stories']:,} stories, {validation['words']:,} words")


def _materialize(
    destination: Path,
    url: str,
    source_size: int,
    source_xet_hash: str,
    *,
    word_limit: int | None = None,
    story_limit: int | None = None,
) -> dict[str, int]:
    temporary = destination.with_name(f".{destination.name}.tmp")
    stories = words = 0
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "project-genesis/0.1",
                "Range": f"bytes=0-{RANGE_BYTES - 1}",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            _verify_response(response, source_size, source_xet_hash)
            lines = io.TextIOWrapper(response, encoding="utf-8")
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for story in _iter_stories(lines):
                    count = len(story.split())
                    stream.write(
                        json.dumps(
                            {"text": story, "source_story": stories + 1},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    stories += 1
                    words += count
                    if (word_limit is not None and words >= word_limit) or (
                        story_limit is not None and stories >= story_limit
                    ):
                        break
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("unable to materialize TinyStories") from error
    if (word_limit is not None and words < word_limit) or (
        story_limit is not None and stories < story_limit
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("TinyStories source prefix was too short")
    os.replace(temporary, destination)
    return {"stories": stories, "words": words}


def _verify_response(response: object, source_size: int, source_xet_hash: str) -> None:
    headers = getattr(response, "headers", None)
    if headers is None:
        raise RuntimeError("TinyStories response has no headers")
    etag = headers.get("ETag", "").strip('"')
    content_range = headers.get("Content-Range", "")
    content_length = headers.get("Content-Length", "")
    size_matches = content_range.endswith(f"/{source_size}") or content_length == str(source_size)
    if etag != source_xet_hash or not size_matches:
        raise RuntimeError("TinyStories source identity did not match the pinned file")


def _iter_stories(lines: Iterable[str]) -> Iterator[str]:
    parts: list[str] = []
    for line in lines:
        if line.strip() == "<|endoftext|>":
            story = "".join(parts).strip()
            parts.clear()
            if story:
                yield story
        else:
            parts.append(line)


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
