import importlib.util
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/prepare_openassistant_conversations.py"
    spec = importlib.util.spec_from_file_location("prepare_openassistant_conversations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conversation_record_selects_the_highest_ranked_english_reply() -> None:
    script = load_script()
    record = script._conversation_record(
        {
            "message_tree_id": "tree-1",
            "prompt": {
                "role": "prompter",
                "lang": "en",
                "text": "How are you?",
                "replies": [
                    {
                        "message_id": "second",
                        "role": "assistant",
                        "lang": "en",
                        "text": "Not the preferred reply.",
                        "rank": 1,
                    },
                    {
                        "message_id": "first",
                        "role": "assistant",
                        "lang": "en",
                        "text": "I am well, thank you!",
                        "rank": 0,
                    },
                ],
            },
        },
        9,
    )

    assert record == {
        "text": "<|user|>\nHow are you?\n<|assistant|>\nI am well, thank you!\n",
        "message_tree_id": "tree-1",
        "assistant_message_id": "first",
        "source_row": 9,
    }


def test_conversation_record_rejects_non_english_prompts() -> None:
    script = load_script()

    assert (
        script._conversation_record(
            {
                "prompt": {
                    "role": "prompter",
                    "lang": "de",
                    "text": "Hallo",
                    "replies": [],
                }
            },
            1,
        )
        is None
    )


def test_training_limit_and_chat_basics_are_deterministic() -> None:
    script = load_script()
    records = [{"source_row": row} for row in range(10)]

    assert script._limit_records(records, 4, 1337) == script._limit_records(records, 4, 1337)
    assert len(script._limit_records(records, 4, 1337)) == 4
    basics = script._basic_records()
    assert len(basics) == 76
    assert basics[0]["text"].startswith("<|user|>\nHi\n<|assistant|>\n")
    assert len(script._basic_validation_records()) == 8
