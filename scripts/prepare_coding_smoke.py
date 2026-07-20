"""Materialize the pinned coding smoke corpus."""

import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = (
    (
        "nanoGPT",
        "https://github.com/karpathy/nanoGPT.git",
        "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
        ROOT / "data/coding-smoke/train/nanoGPT",
    ),
    (
        "minGPT",
        "https://github.com/karpathy/minGPT.git",
        "37baab71b9abea1b76ab957409a1cc2fbfba8a26",
        ROOT / "data/coding-smoke/train/minGPT",
    ),
    (
        "CodeSearchNet",
        "https://github.com/github/CodeSearchNet.git",
        "106e827405c968597da938f6b373d30183918869",
        ROOT / "data/coding-smoke/train/CodeSearchNet",
    ),
    (
        "lit-llama",
        "https://github.com/Lightning-AI/lit-llama.git",
        "2a464de2a1d2f266614d15091d3d7f30330c3ede",
        ROOT / "data/coding-smoke/validation/lit-llama",
    ),
)


def main() -> None:
    """Fetch each repository at its reviewed commit or verify the existing clone."""
    for name, url, revision, destination in REPOSITORIES:
        if destination.exists():
            actual_url = _capture("git", "-C", str(destination), "remote", "get-url", "origin")
            actual_revision = _capture("git", "-C", str(destination), "rev-parse", "HEAD")
            if (actual_url, actual_revision) != (url, revision):
                raise RuntimeError(f"{name} exists at an unexpected origin or revision")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                dir=destination.parent,
                prefix=f".{name}-",
            ) as temporary:
                checkout = Path(temporary) / name
                _run("git", "init", "--quiet", str(checkout))
                _run("git", "-C", str(checkout), "remote", "add", "origin", url)
                _run(
                    "git",
                    "-C",
                    str(checkout),
                    "fetch",
                    "--quiet",
                    "--depth",
                    "1",
                    "origin",
                    revision,
                )
                if name == "CodeSearchNet":
                    archive = Path(temporary) / "CodeSearchNet.tar"
                    _run(
                        "git",
                        "-C",
                        str(checkout),
                        "-c",
                        "core.protectNTFS=false",
                        "archive",
                        "--format=tar",
                        f"--output={archive}",
                        revision,
                        "src",
                        "function_parser",
                        "script",
                        "tests",
                        "README.md",
                        "LICENSE",
                    )
                    with tarfile.open(archive) as stream:
                        stream.extractall(checkout, filter="data")
                    _run("git", "-C", str(checkout), "update-ref", "HEAD", revision)
                else:
                    _run(
                        "git",
                        "-C",
                        str(checkout),
                        "-c",
                        "advice.detachedHead=false",
                        "checkout",
                        "--quiet",
                        "--detach",
                        "FETCH_HEAD",
                    )
                os.replace(checkout, destination)
        print(f"{name}: {revision}")


def _run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _capture(*command: str) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
