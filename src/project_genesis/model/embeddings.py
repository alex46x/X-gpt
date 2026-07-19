"""Token and learned position embedding layers."""

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Map batch-first token IDs to dense vectors."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """Initialize a trainable token lookup table."""
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Embed a ``(batch, sequence)`` integer tensor."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("token_ids must use int32 or int64")
        embedded: Tensor = self.embedding(token_ids)
        return embedded


class LearnedPositionEmbedding(nn.Module):
    """Learn one vector for every supported sequence position."""

    def __init__(self, context_length: int, d_model: int) -> None:
        """Initialize a bounded trainable position lookup table."""
        super().__init__()
        if context_length <= 0 or d_model <= 0:
            raise ValueError("context_length and d_model must be positive")
        self.context_length = context_length
        self.d_model = d_model
        self.embedding = nn.Embedding(context_length, d_model)

    def forward(self, token_embeddings: Tensor, *, offset: int = 0) -> Tensor:
        """Return broadcastable positions for batch-first token embeddings."""
        if token_embeddings.ndim != 3 or token_embeddings.shape[-1] != self.d_model:
            raise ValueError("token_embeddings must have shape (batch, sequence, d_model)")
        if offset < 0:
            raise ValueError("position offset cannot be negative")
        sequence_length = token_embeddings.shape[1]
        if offset + sequence_length > self.context_length:
            raise ValueError(
                f"position range through {offset + sequence_length} exceeds context length "
                f"{self.context_length}"
            )
        positions = torch.arange(
            offset,
            offset + sequence_length,
            device=token_embeddings.device,
        )
        embedded: Tensor = self.embedding(positions).unsqueeze(0)
        return embedded
