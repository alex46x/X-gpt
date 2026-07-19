"""Atomic, deterministic trainer checkpoints."""

import os
import tempfile
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from project_genesis.training.trainer import Trainer

CHECKPOINT_VERSION = 1


def save_checkpoint(path: Path, trainer: Trainer) -> None:
    """Atomically persist model, optimizer, scheduler, scaler, step, and RNG state."""
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "step": trainer.step,
        "microbatches_seen": trainer.microbatches_seen,
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


def load_checkpoint(path: Path, trainer: Trainer) -> None:
    """Restore a trusted Project Genesis checkpoint into a compatible trainer."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    required = {
        "version",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "step",
        "microbatches_seen",
        "cpu_rng_state",
        "cuda_rng_state",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload.get("version") != CHECKPOINT_VERSION
    ):
        raise ValueError("unsupported checkpoint format or version")

    step = payload["step"]
    microbatches_seen = payload["microbatches_seen"]
    if (
        not isinstance(step, int)
        or step < 0
        or not isinstance(microbatches_seen, int)
        or microbatches_seen < 0
    ):
        raise ValueError("checkpoint counters must be non-negative integers")

    trainer.model.load_state_dict(payload["model"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.scheduler.load_state_dict(payload["scheduler"])
    trainer.scaler.load_state_dict(payload["scaler"])
    trainer.step = step
    trainer.microbatches_seen = microbatches_seen
    torch.set_rng_state(cast(Tensor, payload["cpu_rng_state"]))
    cuda_rng_state = payload["cuda_rng_state"]
    if torch.cuda.is_available() and isinstance(cuda_rng_state, list):
        torch.cuda.set_rng_state_all(cuda_rng_state)
