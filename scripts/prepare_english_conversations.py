"""Materialize the pinned Dolly-15k English conversation corpus."""

import argparse
import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path

from project_genesis.chat import Conversation, Message, Role, format_prompt

ROOT = Path(__file__).resolve().parents[1]
DATASET = "databricks/databricks-dolly-15k"
SOURCE_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
SOURCE_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/"
    f"{SOURCE_REVISION}/databricks-dolly-15k.jsonl"
)
SOURCE_SIZE = 13_085_339
SOURCE_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"


def main() -> None:
    """Download, validate, format, and split the conversation corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/english-conversations",
    )
    parser.add_argument("--validation-examples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.validation_examples < 1:
        parser.error("validation-examples must be positive")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "databricks-dolly-15k.jsonl"
    _download(source)
    records = _load_records(source)
    train, validation = _split(records, args.validation_examples, args.seed)
    splits = {"train": train, "validation": validation}
    for name, values in splits.items():
        _write_jsonl(output / f"{name}.jsonl", values)
        print(f"{name}: {len(values):,} examples")

    _write_text(
        output / "manifest.json",
        json.dumps(
            {
                "dataset": DATASET,
                "source": SOURCE_URL,
                "source_revision": SOURCE_REVISION,
                "source_sha256": SOURCE_SHA256,
                "license": "CC-BY-SA-3.0",
                "seed": args.seed,
                "splits": {
                    name: {
                        "examples": len(values),
                        "sha256": _sha256(output / f"{name}.jsonl"),
                    }
                    for name, values in splits.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _download(destination: Path) -> None:
    if destination.exists():
        if destination.stat().st_size == SOURCE_SIZE and _sha256(destination) == SOURCE_SHA256:
            print(f"source: verified {destination}")
            return
        raise RuntimeError(f"existing source failed integrity verification: {destination}")

    temporary = destination.with_suffix(".jsonl.part")
    try:
        request = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "project-genesis/0.1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("unable to download the conversation dataset") from error
    if temporary.stat().st_size != SOURCE_SIZE or _sha256(temporary) != SOURCE_SHA256:
        temporary.unlink()
        raise RuntimeError("downloaded source failed integrity verification")
    os.replace(temporary, destination)


def _load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as lines:
        for source_row, line in enumerate(lines, start=1):
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on source row {source_row}") from error
            record = _conversation_record(value, source_row)
            if record is not None:
                records.append(record)
    if not records:
        raise RuntimeError("conversation source contains no usable records")
    return records


def _conversation_record(value: object, source_row: int) -> dict[str, object] | None:
    if not isinstance(value, dict):
        raise ValueError(f"source row {source_row} must be an object")
    instruction = value.get("instruction")
    context = value.get("context")
    response = value.get("response")
    category = value.get("category")
    if not all(isinstance(item, str) for item in (instruction, context, response, category)):
        raise ValueError(f"source row {source_row} has an invalid schema")
    assert isinstance(instruction, str)
    assert isinstance(context, str)
    assert isinstance(response, str)
    assert isinstance(category, str)
    instruction = instruction.strip()
    response = response.strip()
    if not instruction or not response:
        return None
    context = context.strip()
    prompt = instruction if not context else f"{instruction}\n\nContext:\n{context}"
    return {
        "text": format_prompt(
            Conversation(
                (
                    Message(Role.USER, prompt),
                    Message(Role.ASSISTANT, response),
                )
            )
        ),
        "category": category,
        "source_row": source_row,
    }


def _split(
    records: list[dict[str, object]],
    validation_examples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if validation_examples >= len(records):
        raise ValueError("validation-examples must be smaller than the usable dataset")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_examples])
    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [record for index, record in enumerate(records) if index in validation_indices]
    return train, validation


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


if __name__ == "__main__":
    main()
