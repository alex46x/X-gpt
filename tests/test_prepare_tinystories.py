import importlib.util
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/prepare_tinystories.py"
    spec = importlib.util.spec_from_file_location("prepare_tinystories", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iter_stories_keeps_only_complete_nonempty_stories() -> None:
    script = load_script()

    assert list(
        script._iter_stories(
            ["first line\n", "second line\n", "<|endoftext|>\n", "\n", "<|endoftext|>\n", "partial"]
        )
    ) == ["first line\nsecond line"]
