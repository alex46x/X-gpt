"""Atomic, versioned inference-only model bundles."""

import hashlib
import json
import os
import pickle
import shutil
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

from project_genesis.model import GPTDecoder, load_model_config
from project_genesis.tokenizer import (
    ByteBPETokenizer,
    load_tokenizer,
    save_tokenizer,
)
from project_genesis.utilities import atomic_write_text

BUNDLE_SCHEMA_VERSION = "1.0.0"
MODEL_CONFIG_FILE = "model.yaml"
MODEL_WEIGHTS_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True, slots=True)
class InferenceBundle:
    """Loaded model, tokenizer, and verified bundle identity."""

    model: GPTDecoder
    tokenizer: ByteBPETokenizer
    fingerprint: str
    project_version: str


def save_bundle(
    path: Path,
    model: GPTDecoder,
    tokenizer: ByteBPETokenizer,
) -> str:
    """Atomically create an immutable inference bundle and return its fingerprint."""
    if model.config.vocab_size != len(tokenizer.vocabulary):
        raise ValueError("model and tokenizer vocabulary sizes do not match")
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"bundle destination already exists: {path}")
    temporary = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}-",
        )
    )
    try:
        model_config = yaml.safe_dump(
            {"model": asdict(model.config)},
            sort_keys=True,
        )
        atomic_write_text(temporary / MODEL_CONFIG_FILE, model_config)
        save_tokenizer(tokenizer, temporary / TOKENIZER_FILE)
        _save_weights(temporary / MODEL_WEIGHTS_FILE, model)
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "project_version": version("project-genesis"),
            "model_config_sha256": _sha256(temporary / MODEL_CONFIG_FILE),
            "model_weights_sha256": _sha256(temporary / MODEL_WEIGHTS_FILE),
            "tokenizer_sha256": _sha256(temporary / TOKENIZER_FILE),
            "tokenizer_fingerprint": tokenizer.fingerprint,
        }
        fingerprint = _fingerprint(manifest)
        manifest["bundle_fingerprint"] = fingerprint
        atomic_write_text(
            temporary / MANIFEST_FILE,
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        )
        os.replace(temporary, destination)
        return fingerprint
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_bundle(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> InferenceBundle:
    """Verify and load an inference bundle on the requested device."""
    root = path.expanduser().resolve()
    try:
        manifest = _load_manifest(root / MANIFEST_FILE)
        expected_fingerprint = manifest.pop("bundle_fingerprint")
        if _fingerprint(manifest) != expected_fingerprint:
            raise ValueError("bundle manifest fingerprint does not match")
        checksums = {
            MODEL_CONFIG_FILE: manifest["model_config_sha256"],
            MODEL_WEIGHTS_FILE: manifest["model_weights_sha256"],
            TOKENIZER_FILE: manifest["tokenizer_sha256"],
        }
        for filename, expected in checksums.items():
            if _sha256(root / filename) != expected:
                raise ValueError(f"bundle file checksum does not match: {filename}")

        config = load_model_config(root / MODEL_CONFIG_FILE)
        tokenizer = load_tokenizer(root / TOKENIZER_FILE)
        if config.vocab_size != len(tokenizer.vocabulary):
            raise ValueError("model and tokenizer vocabulary sizes do not match")
        if tokenizer.fingerprint != manifest["tokenizer_fingerprint"]:
            raise ValueError("tokenizer fingerprint does not match bundle manifest")
        model = GPTDecoder(config).to(device)
        state = torch.load(
            root / MODEL_WEIGHTS_FILE,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        return InferenceBundle(
            model=model,
            tokenizer=tokenizer,
            fingerprint=expected_fingerprint,
            project_version=manifest["project_version"],
        )
    except (
        OSError,
        EOFError,
        RuntimeError,
        pickle.UnpicklingError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(f"unable to load inference bundle {path}: {error}") from error


def _save_weights(path: Path, model: GPTDecoder) -> None:
    with path.open("wb") as stream:
        torch.save(model.state_dict(), stream)
        stream.flush()
        os.fsync(stream.fileno())


def _load_manifest(path: Path) -> dict[str, str]:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "project_version",
        "model_config_sha256",
        "model_weights_sha256",
        "tokenizer_sha256",
        "tokenizer_fingerprint",
        "bundle_fingerprint",
    }
    if (
        not isinstance(loaded, dict)
        or set(loaded) != fields
        or not all(isinstance(value, str) and value for value in loaded.values())
    ):
        raise ValueError("bundle manifest fields are invalid")
    if loaded["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported bundle schema version")
    return loaded


def _fingerprint(manifest: dict[str, str]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
