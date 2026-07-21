"""Materialize a pinned English subset of OpenAssistant OASST1."""

import argparse
import gzip
import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path

from project_genesis.chat import Conversation, Message, Role, format_prompt

ROOT = Path(__file__).resolve().parents[1]
DATASET = "OpenAssistant/oasst1"
SOURCE_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
SOURCE_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
    f"{SOURCE_REVISION}/2023-04-12_oasst_ready.trees.jsonl.gz"
)
SOURCE_SIZE = 34_145_252
SOURCE_SHA256 = "2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7"


def main() -> None:
    """Download, validate, format, and split English conversation pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/openassistant-english",
    )
    parser.add_argument("--validation-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.validation_examples < 1:
        parser.error("validation-examples must be positive")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "oasst1-ready.trees.jsonl.gz"
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
                "license": "Apache-2.0",
                "seed": args.seed,
                "filters": {
                    "language": "en",
                    "prompt_max_characters": 1_000,
                    "response_max_characters": 3_000,
                    "reply_selection": "lowest rank, then message ID",
                },
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

    temporary = destination.with_suffix(".gz.part")
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
        raise RuntimeError("unable to download the OpenAssistant dataset") from error
    if temporary.stat().st_size != SOURCE_SIZE or _sha256(temporary) != SOURCE_SHA256:
        temporary.unlink()
        raise RuntimeError("downloaded source failed integrity verification")
    os.replace(temporary, destination)


def _load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        for source_row, line in enumerate(lines, start=1):
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on source row {source_row}") from error
            record = _conversation_record(value, source_row)
            if record is not None:
                records.append(record)
    if not records:
        raise RuntimeError("OpenAssistant source contains no usable English records")
    return records


def _conversation_record(value: object, source_row: int) -> dict[str, object] | None:
    if not isinstance(value, dict) or not isinstance(value.get("prompt"), dict):
        raise ValueError(f"source row {source_row} has an invalid tree schema")
    prompt = value["prompt"]
    assert isinstance(prompt, dict)
    text = prompt.get("text")
    replies = prompt.get("replies")
    if not isinstance(text, str) or not isinstance(replies, list):
        raise ValueError(f"source row {source_row} has an invalid prompt schema")
    text = text.strip()
    if (
        prompt.get("role") != "prompter"
        or prompt.get("lang") != "en"
        or not 1 <= len(text) <= 1_000
    ):
        return None

    candidates = [reply for reply in replies if _usable_reply(reply)]
    if not candidates:
        return None
    reply = min(candidates, key=_reply_order)
    response = reply["text"]
    assert isinstance(response, str)
    response = response.strip()
    return {
        "text": format_prompt(
            Conversation(
                (
                    Message(Role.USER, text),
                    Message(Role.ASSISTANT, response),
                )
            )
        ),
        "message_tree_id": str(value.get("message_tree_id", "")),
        "assistant_message_id": str(reply.get("message_id", "")),
        "source_row": source_row,
    }


def _usable_reply(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    text = value.get("text")
    return (
        value.get("role") == "assistant"
        and value.get("lang") == "en"
        and not value.get("deleted", False)
        and value.get("review_result") is not False
        and isinstance(text, str)
        and 1 <= len(text.strip()) <= 3_000
    )


def _reply_order(reply: dict[object, object]) -> tuple[int, str]:
    rank = reply.get("rank")
    return (rank if isinstance(rank, int) else 2**31, str(reply.get("message_id", "")))


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
