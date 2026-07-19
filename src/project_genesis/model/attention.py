"""Causal multi-head self-attention."""

import math

import torch
from torch import Tensor, nn

type KVCache = tuple[Tensor, Tensor]


class CausalSelfAttention(nn.Module):
    """Apply masked scaled dot-product attention over batch-first states."""

    causal_mask: Tensor

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float,
        *,
        bias: bool = True,
    ) -> None:
        """Initialize QKV and output projections with a causal mask."""
        super().__init__()
        if min(d_model, n_heads, context_length) <= 0:
            raise ValueError("d_model, n_heads, and context_length must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.n_heads = n_heads
        self.context_length = context_length
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer(
            "causal_mask", mask.view(1, 1, context_length, context_length), persistent=False
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Attend to the current and preceding positions only."""
        output, _ = self.forward_cached(inputs)
        return output

    def forward_cached(
        self,
        inputs: Tensor,
        cache: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        """Attend with optional prior keys and values, returning the updated cache."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError("inputs must have shape (batch, sequence, d_model)")
        batch_size, sequence_length, _ = inputs.shape
        past_length = 0 if cache is None else self._validate_cache(cache, inputs)
        total_length = past_length + sequence_length
        if total_length > self.context_length:
            raise ValueError(
                f"cached sequence length {total_length} exceeds context length "
                f"{self.context_length}"
            )

        qkv = self.qkv_projection(inputs).view(
            batch_size,
            sequence_length,
            3,
            self.n_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if cache is not None:
            # ponytail: cache concatenation copies; preallocate if profiling shows pressure.
            key = torch.cat((cache[0], key), dim=2)
            value = torch.cat((cache[1], value), dim=2)
        # ponytail: quadratic attention; swap kernels when profiling justifies SDPA.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[
            :,
            :,
            past_length : past_length + sequence_length,
            :total_length,
        ]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        softmax_dtype = torch.float32 if scores.dtype in {torch.float16, torch.bfloat16} else None
        weights = torch.softmax(scores, dim=-1, dtype=softmax_dtype).to(value.dtype)
        attended = torch.matmul(self.attention_dropout(weights), value)
        combined = (
            attended.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.d_model,
            )
        )
        output: Tensor = self.output_dropout(self.output_projection(combined))
        return output, (key, value)

    def _validate_cache(self, cache: KVCache, inputs: Tensor) -> int:
        key, value = cache
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("cached keys and values must share four-dimensional shape")
        if (
            key.shape[0] != inputs.shape[0]
            or key.shape[1] != self.n_heads
            or key.shape[3] != self.head_dim
        ):
            raise ValueError("cache shape is incompatible with attention inputs")
        if (
            key.dtype != inputs.dtype
            or key.device != inputs.device
            or value.dtype != inputs.dtype
            or value.device != inputs.device
        ):
            raise ValueError("cache dtype and device must match attention inputs")
        return key.shape[2]
