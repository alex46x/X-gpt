"""Token-weighted decoder evaluation."""

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from project_genesis.training import TokenBatch


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate validation quality and throughput metrics."""

    loss: float
    perplexity: float
    token_accuracy: float
    tokens: int
    batches: int
    elapsed_seconds: float
    tokens_per_second: float

    def __post_init__(self) -> None:
        """Validate finite metric values and counts."""
        measurements = (
            self.loss,
            self.perplexity,
            self.token_accuracy,
            self.elapsed_seconds,
            self.tokens_per_second,
        )
        if not all(math.isfinite(value) for value in measurements):
            raise ValueError("evaluation measurements must be finite")
        if self.loss < 0 or self.perplexity < 1:
            raise ValueError("loss must be non-negative and perplexity at least one")
        if not 0 <= self.token_accuracy <= 1:
            raise ValueError("token_accuracy must be in [0, 1]")
        if self.tokens <= 0 or self.batches <= 0:
            raise ValueError("tokens and batches must be positive")
        if self.elapsed_seconds <= 0 or self.tokens_per_second <= 0:
            raise ValueError("timing measurements must be positive")


def evaluate_model(
    model: nn.Module,
    batches: Iterable[TokenBatch],
    *,
    max_batches: int | None = None,
) -> EvaluationResult:
    """Evaluate token-weighted loss, perplexity, accuracy, and throughput."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    device = _model_device(model)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    tokens = 0
    batch_count = 0
    _synchronize(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for inputs, targets in batches:
                if max_batches is not None and batch_count >= max_batches:
                    break
                _validate_batch(inputs, targets)
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = cast(Tensor, model(inputs))
                flattened_targets = targets.reshape(-1)
                total_loss += functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    flattened_targets,
                    reduction="sum",
                ).item()
                correct += int(
                    (logits.argmax(dim=-1).reshape(-1) == flattened_targets).sum().item()
                )
                tokens += flattened_targets.numel()
                batch_count += 1
        _synchronize(device)
    finally:
        model.train(was_training)

    elapsed = time.perf_counter() - started
    if not tokens:
        raise ValueError("evaluation requires at least one non-empty batch")
    mean_loss = total_loss / tokens
    return EvaluationResult(
        loss=mean_loss,
        perplexity=math.exp(mean_loss),
        token_accuracy=correct / tokens,
        tokens=tokens,
        batches=batch_count,
        elapsed_seconds=elapsed,
        tokens_per_second=tokens / elapsed,
    )


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_batch(inputs: Tensor, targets: Tensor) -> None:
    if inputs.ndim != 2 or inputs.shape != targets.shape or inputs.numel() == 0:
        raise ValueError("inputs and targets must share non-empty shape (batch, sequence)")
    if inputs.dtype not in {torch.int32, torch.int64} or targets.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise TypeError("inputs and targets must contain integer token IDs")
