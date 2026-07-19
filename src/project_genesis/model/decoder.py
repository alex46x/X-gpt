"""GPT-style decoder-only language model."""

import math

from torch import Tensor, nn

from project_genesis.model.block import TransformerBlock
from project_genesis.model.config import ModelConfig
from project_genesis.model.embeddings import LearnedPositionEmbedding, TokenEmbedding
from project_genesis.model.normalization import LayerNorm


class GPTDecoder(nn.Module):
    """Map batch-first token IDs to next-token logits."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize embeddings, decoder blocks, normalization, and output head."""
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.position_embedding = LearnedPositionEmbedding(
            config.context_length,
            config.d_model,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = LayerNorm(
            config.d_model,
            config.layer_norm_epsilon,
            bias=config.bias,
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        residual_std = config.initializer_range / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            if not isinstance(block, TransformerBlock):
                raise TypeError("decoder blocks must be TransformerBlock instances")
            nn.init.normal_(block.attention.output_projection.weight, std=residual_std)
            nn.init.normal_(block.feed_forward.output_projection.weight, std=residual_std)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return logits with shape ``(batch, sequence, vocab_size)``."""
        if token_ids.ndim == 2 and token_ids.shape[1] == 0:
            raise ValueError("token sequence must not be empty")
        token_states: Tensor = self.token_embedding(token_ids)
        position_states: Tensor = self.position_embedding(token_states)
        hidden: Tensor = self.embedding_dropout(token_states + position_states)
        for block in self.blocks:
            hidden = block(hidden)
        normalized: Tensor = self.final_norm(hidden)
        logits: Tensor = self.lm_head(normalized)
        return logits

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count unique parameters, optionally excluding frozen parameters."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
