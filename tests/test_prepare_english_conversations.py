import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/prepare_english_conversations.py"
    spec = importlib.util.spec_from_file_location("prepare_english_conversations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conversation_record_uses_chat_prompt_contract() -> None:
    script = load_script()

    record = script._conversation_record(
        {
            "instruction": "How are you?",
            "context": "Reply warmly.",
            "response": "I am doing well, thank you!",
            "category": "open_qa",
        },
        7,
    )

    assert record == {
        "text": (
            "<|user|>\nHow are you?\n\nContext:\nReply warmly.\n"
            "<|assistant|>\nI am doing well, thank you!\n"
        ),
        "category": "open_qa",
        "source_row": 7,
    }


def test_split_is_deterministic_and_rejects_an_empty_training_split() -> None:
    script = load_script()
    records = [{"source_row": row} for row in range(5)]

    assert script._split(records, 2, 1337) == script._split(records, 2, 1337)
    with pytest.raises(ValueError, match="smaller"):
        script._split(records, len(records), 1337)
