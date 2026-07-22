"""Materialize a pinned English subset of an OpenAssistant release."""

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
SOURCES = {
    "oasst1": {
        "dataset": "OpenAssistant/oasst1",
        "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "filename": "2023-04-12_oasst_ready.trees.jsonl.gz",
        "size": 34_145_252,
        "sha256": "2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7",
        "output": "openassistant-english",
    },
    "oasst2": {
        "dataset": "OpenAssistant/oasst2",
        "revision": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
        "filename": "2023-11-05_oasst2_ready.trees.jsonl.gz",
        "size": 54_370_156,
        "sha256": "7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b",
        "output": "openassistant2-english",
    },
}


def main() -> None:
    """Download, validate, format, and split English conversation pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=tuple(SOURCES), default="oasst1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-examples", type=int, default=300)
    parser.add_argument("--training-examples", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.validation_examples < 1:
        parser.error("validation-examples must be positive")
    if args.training_examples is not None and args.training_examples < 1:
        parser.error("training-examples must be positive")

    selected = SOURCES[args.release]
    dataset = str(selected["dataset"])
    revision = str(selected["revision"])
    filename = str(selected["filename"])
    source_size = selected["size"]
    assert isinstance(source_size, int)
    source_url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{filename}"
    output = (args.output or ROOT / f"data/{selected['output']}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / filename
    _download(source, source_url, source_size, str(selected["sha256"]))
    records = _load_records(source)
    train, validation = _split(records, args.validation_examples, args.seed)
    train = _limit_records(train, args.training_examples, args.seed)
    splits = {"train": train, "validation": validation}
    for name, values in splits.items():
        _write_jsonl(output / f"{name}.jsonl", values)
        print(f"{name}: {len(values):,} examples")
    basics = _basic_records() if args.release == "oasst2" else []
    basic_validation = _basic_validation_records() if basics else []
    if basics:
        _write_jsonl(output / "basics.jsonl", basics)
        _write_jsonl(output / "basics-validation.jsonl", basic_validation)
        print(f"basics: {len(basics):,} examples")

    _write_text(
        output / "manifest.json",
        json.dumps(
            {
                "dataset": dataset,
                "source": source_url,
                "source_revision": revision,
                "source_sha256": selected["sha256"],
                "license": "Apache-2.0",
                "seed": args.seed,
                "filters": {
                    "language": "en",
                    "prompt_max_characters": 1_000,
                    "response_max_characters": 3_000,
                    "reply_selection": "lowest rank, then message ID",
                },
                "chat_basics": (
                    {
                        "license": "MIT",
                        "train_examples": len(basics),
                        "train_sha256": _sha256(output / "basics.jsonl"),
                        "validation_examples": len(basic_validation),
                        "validation_sha256": _sha256(output / "basics-validation.jsonl"),
                    }
                    if basics
                    else None
                ),
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


def _download(destination: Path, url: str, size: int, sha256: str) -> None:
    if destination.exists():
        if destination.stat().st_size == size and _sha256(destination) == sha256:
            print(f"source: verified {destination}")
            return
        raise RuntimeError(f"existing source failed integrity verification: {destination}")

    temporary = destination.with_suffix(".gz.part")
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "project-genesis/0.1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("unable to download the OpenAssistant dataset") from error
    if temporary.stat().st_size != size or _sha256(temporary) != sha256:
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


def _limit_records(
    records: list[dict[str, object]],
    limit: int | None,
    seed: int,
) -> list[dict[str, object]]:
    if limit is None or limit >= len(records):
        return records
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    selected = set(indices[:limit])
    return [record for index, record in enumerate(records) if index in selected]


def _basic_records() -> list[dict[str, object]]:
    groups = (
        (
            ("Hi", "Hello", "Hello there", "Hey", "Hi there", "Good morning", "Good evening"),
            (
                "Hello! How can I help you today?",
                "Hi! What can I help you with?",
                "Hello! It is nice to meet you.",
                "Hi there! How may I assist you?",
            ),
        ),
        (
            ("How are you?", "How are you doing?", "Are you well?", "How is it going?"),
            (
                "I am doing well, thank you! How are you?",
                "I am fine, thank you. How can I help you today?",
                "I am ready to help. How are you doing?",
            ),
        ),
        (
            ("What is your name?", "Who are you?", "Can you introduce yourself?"),
            (
                "I am Project Genesis, a small locally trained AI assistant.",
                "My name is Project Genesis. How can I help you?",
            ),
        ),
        (
            ("Can you help me?", "I need help", "Could you assist me?", "Please help me"),
            (
                "Of course. Tell me what you need help with.",
                "Yes, I will do my best to help. What would you like to know?",
                "Certainly. Please describe the problem.",
            ),
        ),
        (
            ("Thank you", "Thanks", "Thank you for your help"),
            ("You are welcome!", "Happy to help!", "You are very welcome."),
        ),
        (
            ("Goodbye", "Bye", "See you later"),
            ("Goodbye!", "See you later!", "Take care!"),
        ),
    )
    records: list[dict[str, object]] = []
    for prompts, responses in groups:
        for prompt in prompts:
            for response in responses:
                records.append(
                    {
                        "text": format_prompt(
                            Conversation(
                                (
                                    Message(Role.USER, prompt),
                                    Message(Role.ASSISTANT, response),
                                )
                            )
                        ),
                        "category": "chat_basics",
                    }
                )
    return records


def _basic_validation_records() -> list[dict[str, object]]:
    pairs = (
        ("Hi!", "Hello! How can I help you today?"),
        ("Hey there!", "Hi there! What can I help you with?"),
        ("How have you been?", "I am doing well, thank you! How are you?"),
        ("How do you feel today?", "I am fine, thank you. How can I help you today?"),
        ("Tell me your name", "My name is Project Genesis. How can I help you?"),
        ("Would you help me?", "Of course. Tell me what you need help with."),
        ("Many thanks", "You are very welcome."),
        ("See you soon", "Take care!"),
    )
    return [
        {
            "text": format_prompt(
                Conversation(
                    (
                        Message(Role.USER, prompt),
                        Message(Role.ASSISTANT, response),
                    )
                )
            ),
            "category": "chat_basics_validation",
        }
        for prompt, response in pairs
    ]


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
