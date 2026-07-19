"""Custom layer normalization."""

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """Normalize the final dimension with learned scale and optional bias."""

    def __init__(self, d_model: int, epsilon: float, *, bias: bool = True) -> None:
        """Initialize learned normalization parameters."""
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.d_model = d_model
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, inputs: Tensor) -> Tensor:
        """Normalize a tensor whose final dimension is ``d_model``."""
        if inputs.shape[-1] != self.d_model:
            raise ValueError(f"final dimension must be {self.d_model}")
        statistics = inputs.float() if inputs.dtype in {torch.float16, torch.bfloat16} else inputs
        centered = statistics - statistics.mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(
            centered.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        output = normalized.to(inputs.dtype) * self.weight.to(inputs.dtype)
        return output if self.bias is None else output + self.bias.to(inputs.dtype)
