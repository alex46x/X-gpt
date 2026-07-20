"""Pre-normalization decoder transformer block."""

from torch import Tensor, nn

from project_genesis.model.attention import CausalSelfAttention, KVCache
from project_genesis.model.config import ModelConfig
from project_genesis.model.feed_forward import FeedForward
from project_genesis.model.normalization import build_normalization
from project_genesis.model.residual import residual_add


class TransformerBlock(nn.Module):
    """Compose causal attention and feed-forward residual sublayers."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize one pre-normalization decoder block."""
        super().__init__()
        self.attention_norm = build_normalization(
            config.d_model,
            config.layer_norm_epsilon,
            config.normalization,
            bias=config.bias,
        )
        self.attention = CausalSelfAttention(
            config.d_model,
            config.n_heads,
            config.context_length,
            config.dropout,
            bias=config.bias,
            n_kv_heads=config.kv_heads,
            rope_theta=config.rope_theta if config.position_encoding == "rope" else None,
            use_sdpa=config.use_sdpa,
        )
        self.feed_forward_norm = build_normalization(
            config.d_model,
            config.layer_norm_epsilon,
            config.normalization,
            bias=config.bias,
        )
        self.feed_forward = FeedForward(
            config.d_model,
            config.d_ff,
            config.dropout,
            bias=config.bias,
            activation=config.feed_forward,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Transform hidden states while preserving their shape."""
        output, _ = self.forward_cached(inputs)
        return output

    def forward_cached(
        self,
        inputs: Tensor,
        cache: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        """Transform hidden states and return updated attention keys and values."""
        attention_input: Tensor = self.attention_norm(inputs)
        attention_output, updated_cache = self.attention.forward_cached(
            attention_input,
            cache,
        )
        hidden = residual_add(inputs, attention_output)
        feed_forward_input: Tensor = self.feed_forward_norm(hidden)
        feed_forward_output: Tensor = self.feed_forward(feed_forward_input)
        return residual_add(hidden, feed_forward_output), updated_cache
