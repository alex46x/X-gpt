"""Native PyTorch optimizer and learning-rate scheduling."""

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from project_genesis.training.config import TrainingConfig


def create_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Create AdamW with decay limited to matrix-shaped parameters."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("model has no trainable parameters")
    return AdamW(
        (
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )


def create_scheduler(optimizer: Optimizer, config: TrainingConfig) -> LRScheduler:
    """Create optional linear warmup followed by cosine decay."""
    decay_steps = config.max_steps - config.warmup_steps
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=config.learning_rate * config.min_learning_rate_ratio,
    )
    if not config.warmup_steps:
        return cosine
    warmup = LinearLR(
        optimizer,
        start_factor=1 / config.warmup_steps,
        total_iters=config.warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config.warmup_steps],
    )
