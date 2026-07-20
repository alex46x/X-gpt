"""Materialize the pinned coding smoke corpus."""

import os
import subprocess
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
