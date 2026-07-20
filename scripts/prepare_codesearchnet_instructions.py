"""Materialize a bounded CodeSearchNet Python instruction dataset."""

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from project_genesis.chat import Conversation, Message, Role, format_prompt

ROOT = Path(__file__).resolve().parents[1]
DATASET = "github/CodeSearchNet"
ARCHIVE_URL = "https://zenodo.org/api/records/7857872/files/python.zip/content"
ARCHIVE_SIZE = 940_909_997
ARCHIVE_MD5 = "07b49dd01fbac894fbdae22da6462e4f"
ARCHIVE_DOI = "10.5281/zenodo.7857872"


def main() -> None:
    """Download, filter, and atomically publish deterministic instruction splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/codesearchnet-python",
    )
    parser.add_argument("--train-examples", type=int, default=12_000)
    parser.add_argument("--validation-examples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.train_examples < 1 or args.validation_examples < 1:
        parser.error("example counts must be positive")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "python.zip"
    _download_archive(archive)
    splits = {
        "train": _materialize_split(archive, "train", args.train_examples, args.seed),
        "validation": _materialize_split(
            archive,
            "validation",
            args.validation_examples,
            args.seed + 1,
        ),
    }
    for split, records in splits.items():
        _write_jsonl(output / f"{split}.jsonl", records)
        print(f"{split}: {len(records):,} examples")

    manifest = {
        "dataset": DATASET,
        "source": ARCHIVE_URL,
        "archive_doi": ARCHIVE_DOI,
        "archive_md5": ARCHIVE_MD5,
        "license": "mixed upstream licenses; retain each func_code_url for attribution",
        "seed": args.seed,
        "filters": {
            "documentation_characters": [20, 600],
            "code_characters": [20, 2400],
        },
        "splits": {
            split: {
                "examples": len(records),
                "sha256": _sha256(output / f"{split}.jsonl"),
            }
            for split, records in splits.items()
        },
    }
    _write_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")


def _materialize_split(
    archive_path: Path,
    split: str,
    target: int,
    seed: int,
) -> list[dict[str, object]]:
    archive_split = "valid" if split == "validation" else split
    randomizer = random.Random(seed)
    records: list[dict[str, object]] = []
    usable = 0
    row_index = 0
    with ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if f"/jsonl/{archive_split}/" in name and name.endswith(".jsonl.gz")
        )
        if not members:
            raise RuntimeError(f"archive has no {archive_split!r} JSONL members")
        for member in members:
            with (
                archive.open(member) as compressed,
                gzip.GzipFile(fileobj=compressed) as uncompressed,
                io.TextIOWrapper(uncompressed, encoding="utf-8") as lines,
            ):
                for line in lines:
                    row_index += 1
                    value = json.loads(line)
                    record = _instruction_record({"row_idx": row_index, "row": value})
                    if record is None:
                        continue
                    usable += 1
                    if len(records) < target:
                        records.append(record)
                    else:
                        replacement = randomizer.randrange(usable)
                        if replacement < target:
                            records[replacement] = record
                    if row_index % 100_000 == 0:
                        print(f"{split}: scanned {row_index:,} rows")
    if len(records) < target:
        raise RuntimeError(f"only {len(records):,} of {target:,} usable {split} examples found")
    records.sort(key=lambda record: int(record["source_row"]))
    return records


def _download_archive(destination: Path) -> None:
    if destination.exists():
        if destination.stat().st_size == ARCHIVE_SIZE and _md5(destination) == ARCHIVE_MD5:
            print(f"archive: verified {destination}")
            return
        raise RuntimeError(f"existing archive failed integrity verification: {destination}")

    temporary = destination.with_suffix(".zip.part")
    downloaded = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={
            "User-Agent": "project-genesis/0.1",
            **({"Range": f"bytes={downloaded}-"} if downloaded else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            append = downloaded > 0 and response.status == 206
            if not append:
                downloaded = 0
            with temporary.open("ab" if append else "wb") as stream:
                while chunk := response.read(4 * 1024 * 1024):
                    stream.write(chunk)
                    downloaded += len(chunk)
                    print(
                        f"\rarchive: {downloaded / 1024**2:,.0f} / "
                        f"{ARCHIVE_SIZE / 1024**2:,.0f} MiB",
                        end="",
                        flush=True,
                    )
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise RuntimeError(f"download interrupted; rerun to resume from {temporary}") from error
    print()
    if temporary.stat().st_size != ARCHIVE_SIZE or _md5(temporary) != ARCHIVE_MD5:
        raise RuntimeError("downloaded archive failed published size or MD5 verification")
    os.replace(temporary, destination)


def _instruction_record(item: object) -> dict[str, object] | None:
    if not isinstance(item, dict) or item.get("truncated_cells"):
        return None
    row = item.get("row")
    row_index = item.get("row_idx")
    if not isinstance(row, dict) or not isinstance(row_index, int):
        return None
    documentation = row.get("func_documentation_string", row.get("docstring"))
    code = row.get("func_code_string", row.get("code"))
    if not isinstance(documentation, str) or not isinstance(code, str):
        return None
    documentation = documentation.strip()
    code = code.strip()
    if not 20 <= len(documentation) <= 600 or not 20 <= len(code) <= 2400:
        return None

    prompt = f"Write a Python function that satisfies this requirement:\n\n{documentation}"
    text = format_prompt(
        Conversation(
            (
                Message(Role.USER, prompt),
                Message(Role.ASSISTANT, code),
            )
        )
    )
    return {
        "text": text,
        "repository": row.get("repository_name", row.get("repo", "")),
        "path": row.get("func_path_in_repository", row.get("path", "")),
        "url": row.get("func_code_url", row.get("url", "")),
        "language": "python",
        "source_row": row_index,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    )
    _write_text(path, content)


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


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
