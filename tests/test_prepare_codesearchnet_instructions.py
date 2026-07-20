import importlib.util
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/prepare_codesearchnet_instructions.py"
    spec = importlib.util.spec_from_file_location("prepare_codesearchnet_instructions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_instruction_record_uses_chat_prompt_contract() -> None:
    script = load_script()
    record = script._instruction_record(
        {
            "row_idx": 42,
            "row": {
                "repository_name": "owner/repo",
                "func_path_in_repository": "maths.py",
                "func_documentation_string": "Return the sum of two numbers.",
                "func_code_string": "def add(a, b):\n    return a + b",
                "func_code_url": "https://example.test/maths.py",
            },
            "truncated_cells": [],
        }
    )

    assert record == {
        "text": (
            "<|user|>\nWrite a Python function that satisfies this requirement:\n\n"
            "Return the sum of two numbers.\n"
            "<|assistant|>\ndef add(a, b):\n    return a + b\n"
        ),
        "repository": "owner/repo",
        "path": "maths.py",
        "url": "https://example.test/maths.py",
        "language": "python",
        "source_row": 42,
    }


def test_instruction_record_rejects_truncated_and_oversized_rows() -> None:
    script = load_script()
    base = {
        "row_idx": 1,
        "row": {
            "func_documentation_string": "A sufficiently long requirement.",
            "func_code_string": "def function():\n    return True",
        },
        "truncated_cells": [],
    }

    assert script._instruction_record({**base, "truncated_cells": ["func_code_string"]}) is None
    assert (
        script._instruction_record(
            {
                **base,
                "row": {
                    **base["row"],
                    "func_code_string": "x" * 2401,
                },
            }
        )
        is None
    )


def test_instruction_record_accepts_original_archive_schema() -> None:
    script = load_script()
    record = script._instruction_record(
        {
            "row_idx": 7,
            "row": {
                "repo": "owner/repo",
                "path": "module.py",
                "docstring": "Return whether a value is valid.",
                "code": "def is_valid(value):\n    return bool(value)",
                "url": "https://example.test/module.py",
            },
        }
    )

    assert record is not None
    assert record["repository"] == "owner/repo"
    assert record["source_row"] == 7
