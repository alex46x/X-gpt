"""Decoder language-model training step."""

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from project_genesis.training.config import Precision, TrainingConfig
from project_genesis.training.data import TokenBatch
from project_genesis.training.optimization import create_optimizer, create_scheduler


class Trainer:
    """Own model optimization state and execute accumulated training steps."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        """Move the model to its device and initialize optimization state."""
        self.config = config
        self.device = torch.device(device)
        if config.precision is Precision.FLOAT16 and self.device.type != "cuda":
            raise ValueError("float16 training requires a CUDA device")
        self.model = model.to(self.device)
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.precision is Precision.FLOAT16,
        )
        self.step = 0
        self.microbatches_seen = 0

    def train_step(self, microbatches: Sequence[TokenBatch]) -> float:
        """Run one optimizer step and return the mean unscaled loss."""
        if self.step >= self.config.max_steps:
            raise RuntimeError("maximum training steps reached")
        if len(microbatches) != self.config.gradient_accumulation_steps:
            raise ValueError("microbatch count must equal gradient_accumulation_steps")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for inputs, targets in microbatches:
            self._validate_batch(inputs, targets)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            with self._autocast():
                logits = cast(Tensor, self.model(inputs))
                loss = functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("training loss is not finite")
            total_loss += loss.detach().item()
            torch.autograd.backward(
                self.scaler.scale(
                    loss / self.config.gradient_accumulation_steps,
                )
            )

        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_gradient_norm,
            error_if_nonfinite=True,
        )
        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.microbatches_seen += len(microbatches)
        if not self.scaler.is_enabled() or self.scaler.get_scale() >= previous_scale:
            self.scheduler.step()
            self.step += 1
        return total_loss / len(microbatches)

    def _autocast(self) -> AbstractContextManager[None]:
        if self.config.precision is Precision.FLOAT32:
            return nullcontext()
        dtype = torch.bfloat16 if self.config.precision is Precision.BFLOAT16 else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    @staticmethod
    def _validate_batch(inputs: Tensor, targets: Tensor) -> None:
        if inputs.ndim != 2 or inputs.shape != targets.shape:
            raise ValueError("inputs and targets must share shape (batch, sequence)")
        if inputs.dtype not in {torch.int32, torch.int64} or targets.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise TypeError("inputs and targets must contain integer token IDs")


def seed_training(seed: int) -> None:
    """Seed PyTorch CPU and available CUDA generators."""
    if seed < 0:
        raise ValueError("seed cannot be negative")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
