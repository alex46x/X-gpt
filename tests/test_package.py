from importlib.metadata import metadata, version

import project_genesis


def test_package_is_importable_with_expected_metadata() -> None:
    assert project_genesis.__name__ == "project_genesis"
    assert metadata("project-genesis")["Name"] == "project-genesis"
    assert version("project-genesis") == "0.1.0"
