"""Causal multi-head self-attention."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

type KVCache = tuple[Tensor, Tensor]


class CausalSelfAttention(nn.Module):
    """Apply masked scaled dot-product attention over batch-first states."""

    rope_frequencies: Tensor

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float,
        *,
        bias: bool = True,
        n_kv_heads: int | None = None,
        rope_theta: float | None = None,
        use_sdpa: bool = False,
    ) -> None:
        """Initialize QKV and output projections with a causal mask."""
        super().__init__()
        if min(d_model, n_heads, context_length) <= 0:
            raise ValueError("d_model, n_heads, and context_length must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if kv_heads <= 0 or n_heads % kv_heads:
            raise ValueError("n_kv_heads must be positive and divide n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if rope_theta is not None and (rope_theta <= 0 or d_model // n_heads % 2):
            raise ValueError("RoPE requires a positive theta and even head dimension")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = kv_heads
        self.context_length = context_length
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_sdpa = use_sdpa
        qkv_width = (n_heads + 2 * kv_heads) * self.head_dim
        self.qkv_projection = nn.Linear(d_model, qkv_width, bias=bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        frequencies = (
            torch.empty(0)
            if rope_theta is None
            else rope_theta
            ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("rope_frequencies", frequencies, persistent=False)

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

        query_width = self.n_heads * self.head_dim
        kv_width = self.n_kv_heads * self.head_dim
        query, key, value = self.qkv_projection(inputs).split(
            (query_width, kv_width, kv_width),
            dim=-1,
        )
        query = query.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.n_kv_heads, self.head_dim).transpose(
            1, 2
        )
        if self.rope_frequencies.numel():
            query = self._apply_rope(query, offset=past_length)
            key = self._apply_rope(key, offset=past_length)
        if cache is not None:
            # ponytail: cache concatenation copies; preallocate if profiling shows pressure.
            key = torch.cat((cache[0], key), dim=2)
            value = torch.cat((cache[1], value), dim=2)
        if self.use_sdpa:
            mask = (
                None
                if past_length == 0
                else self._causal_mask(
                    sequence_length,
                    total_length,
                    past_length,
                    inputs.device,
                )
            )
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=mask is None,
                scale=self.scale,
                enable_gqa=self.n_heads != self.n_kv_heads,
            )
        else:
            if self.n_heads != self.n_kv_heads:
                repeats = self.n_heads // self.n_kv_heads
                attended_key = key.repeat_interleave(repeats, dim=1)
                attended_value = value.repeat_interleave(repeats, dim=1)
            else:
                attended_key, attended_value = key, value
            scores = torch.matmul(query, attended_key.transpose(-2, -1)) * self.scale
            mask = self._causal_mask(
                sequence_length,
                total_length,
                past_length,
                inputs.device,
            )
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            softmax_dtype = (
                torch.float32 if scores.dtype in {torch.float16, torch.bfloat16} else None
            )
            weights = torch.softmax(scores, dim=-1, dtype=softmax_dtype).to(value.dtype)
            attended = torch.matmul(self.attention_dropout(weights), attended_value)
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

    def _apply_rope(self, inputs: Tensor, *, offset: int) -> Tensor:
        positions = torch.arange(
            offset,
            offset + inputs.shape[2],
            device=inputs.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, self.rope_frequencies.float())
        cosine = angles.cos().to(inputs.dtype).view(1, 1, inputs.shape[2], -1)
        sine = angles.sin().to(inputs.dtype).view(1, 1, inputs.shape[2], -1)
        pairs = inputs.view(*inputs.shape[:-1], -1, 2)
        first, second = pairs.unbind(-1)
        return torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine),
            dim=-1,
        ).flatten(-2)

    @staticmethod
    def _causal_mask(
        sequence_length: int,
        total_length: int,
        past_length: int,
        device: torch.device,
    ) -> Tensor:
        queries = torch.arange(
            past_length,
            past_length + sequence_length,
            device=device,
        ).view(-1, 1)
        keys = torch.arange(total_length, device=device).view(1, -1)
        return (keys <= queries).view(1, 1, sequence_length, total_length)

    def _validate_cache(self, cache: KVCache, inputs: Tensor) -> int:
        key, value = cache
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("cached keys and values must share four-dimensional shape")
        if (
            key.shape[0] != inputs.shape[0]
            or key.shape[1] != self.n_kv_heads
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
