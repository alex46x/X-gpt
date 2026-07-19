"""Pre-normalization decoder transformer block."""

from torch import Tensor, nn

from project_genesis.model.attention import CausalSelfAttention
from project_genesis.model.config import ModelConfig
from project_genesis.model.feed_forward import FeedForward
from project_genesis.model.normalization import LayerNorm
from project_genesis.model.residual import residual_add


class TransformerBlock(nn.Module):
    """Compose causal attention and feed-forward residual sublayers."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize one pre-normalization decoder block."""
        super().__init__()
        self.attention_norm = LayerNorm(
            config.d_model,
            config.layer_norm_epsilon,
            bias=config.bias,
        )
        self.attention = CausalSelfAttention(
            config.d_model,
            config.n_heads,
            config.context_length,
            config.dropout,
            bias=config.bias,
        )
        self.feed_forward_norm = LayerNorm(
            config.d_model,
            config.layer_norm_epsilon,
            bias=config.bias,
        )
        self.feed_forward = FeedForward(
            config.d_model,
            config.d_ff,
            config.dropout,
            bias=config.bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Transform hidden states while preserving their shape."""
        attention_input: Tensor = self.attention_norm(inputs)
        attention_output: Tensor = self.attention(attention_input)
        hidden = residual_add(inputs, attention_output)
        feed_forward_input: Tensor = self.feed_forward_norm(hidden)
        feed_forward_output: Tensor = self.feed_forward(feed_forward_input)
        return residual_add(hidden, feed_forward_output)
