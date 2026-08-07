"""Atomic, versioned trainer checkpoints with legacy loading."""

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from project_genesis.training.trainer import Trainer

CHECKPOINT_VERSION = 2
LEGACY_CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: Path,
    trainer: Trainer,
    *,
    epoch: int = 0,
    configuration: Mapping[str, object] | None = None,
    tokenizer_fingerprint: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Atomically persist full training state and reproducibility metadata."""
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "scaler_state_dict": trainer.scaler.state_dict(),
        "training_step": trainer.step,
        "epoch": epoch,
        "microbatches_seen": trainer.microbatches_seen,
        "configuration": dict(configuration or {}),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "metadata": dict(metadata or {}),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}-",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path, trainer: Trainer) -> dict[str, object]:
    """Restore a version-1 or version-2 checkpoint and return its metadata."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    version = payload.get("version")
    if version == LEGACY_CHECKPOINT_VERSION:
        fields = {
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "step",
            "microbatches_seen",
            "cpu_rng_state",
            "cuda_rng_state",
            "version",
        }
        if set(payload) != fields:
            raise ValueError("unsupported legacy checkpoint format")
        model_state = payload["model"]
        optimizer_state = payload["optimizer"]
        scheduler_state = payload["scheduler"]
        scaler_state = payload["scaler"]
        step = payload["step"]
        metadata: dict[str, object] = {}
        epoch = 0
        configuration: dict[str, object] = {}
        tokenizer_fingerprint = None
    elif version == CHECKPOINT_VERSION:
        fields = {
            "version",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "training_step",
            "epoch",
            "microbatches_seen",
            "configuration",
            "tokenizer_fingerprint",
            "metadata",
            "cpu_rng_state",
            "cuda_rng_state",
        }
        if set(payload) != fields:
            raise ValueError("unsupported checkpoint format")
        model_state = payload["model_state_dict"]
        optimizer_state = payload["optimizer_state_dict"]
        scheduler_state = payload["scheduler_state_dict"]
        scaler_state = payload["scaler_state_dict"]
        step = payload["training_step"]
        epoch = payload["epoch"]
        configuration = payload["configuration"]
        metadata = payload["metadata"]
        tokenizer_fingerprint = payload["tokenizer_fingerprint"]
    else:
        raise ValueError("unsupported checkpoint version")

    if (
        not isinstance(step, int)
        or step < 0
        or not isinstance(epoch, int)
        or epoch < 0
        or not isinstance(payload["microbatches_seen"], int)
        or payload["microbatches_seen"] < 0
    ):
        raise ValueError("checkpoint counters must be non-negative integers")
    if not isinstance(configuration, dict) or not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be mappings")
    if tokenizer_fingerprint is not None and not isinstance(tokenizer_fingerprint, str):
        raise ValueError("checkpoint tokenizer fingerprint must be a string or null")

    trainer.model.load_state_dict(model_state)
    trainer.optimizer.load_state_dict(optimizer_state)
    trainer.scheduler.load_state_dict(scheduler_state)
    trainer.scaler.load_state_dict(scaler_state)
    trainer.step = step
    trainer.microbatches_seen = payload["microbatches_seen"]
    trainer.epoch = epoch
    torch.set_rng_state(cast(Tensor, payload["cpu_rng_state"]))
    cuda_rng_state = payload["cuda_rng_state"]
    if torch.cuda.is_available() and isinstance(cuda_rng_state, list):
        torch.cuda.set_rng_state_all(cuda_rng_state)
    return {
        "version": version,
        "epoch": epoch,
        "configuration": configuration,
        "metadata": metadata,
        "tokenizer_fingerprint": tokenizer_fingerprint,
    }
