import pytest
import torch

from project_genesis.model import LearnedPositionEmbedding, TokenEmbedding


def test_token_and_position_embeddings_have_batch_first_shapes_and_gradients() -> None:
    token_embedding = TokenEmbedding(vocab_size=32, d_model=8)
    position_embedding = LearnedPositionEmbedding(context_length=16, d_model=8)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    tokens = token_embedding(token_ids)
    positions = position_embedding(tokens)
    output = tokens + positions
    output.sum().backward()

    assert tokens.shape == (2, 3, 8)
    assert positions.shape == (1, 3, 8)
    assert token_embedding.embedding.weight.grad is not None
    assert position_embedding.embedding.weight.grad is not None


def test_embeddings_reject_invalid_inputs_and_context_overflow() -> None:
    tokens = TokenEmbedding(vocab_size=8, d_model=4)
    positions = LearnedPositionEmbedding(context_length=2, d_model=4)

    with pytest.raises(TypeError, match="int32 or int64"):
        tokens(torch.ones(1, 2))
    with pytest.raises(ValueError, match="exceeds context"):
        positions(torch.zeros(1, 3, 4))
