import pytest
import torch

from project_genesis.model import CausalSelfAttention


def attention(dropout: float = 0.0) -> CausalSelfAttention:
    return CausalSelfAttention(
        d_model=8,
        n_heads=2,
        context_length=8,
        dropout=dropout,
    )


def test_attention_preserves_shape_and_backpropagates() -> None:
    module = attention()
    inputs = torch.randn(2, 5, 8, requires_grad=True)

    output = module(inputs)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None
    assert module.qkv_projection.weight.grad is not None
    assert "causal_mask" not in module.state_dict()


def test_attention_is_causal() -> None:
    torch.manual_seed(7)
    module = attention()
    module.eval()
    original = torch.randn(1, 6, 8)
    changed = original.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:])

    original_output = module(original)
    changed_output = module(changed)

    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])
    assert not torch.allclose(original_output[:, 3:], changed_output[:, 3:])


def test_cached_attention_matches_full_attention() -> None:
    module = attention().eval()
    inputs = torch.randn(2, 6, 8)

    full = module(inputs)
    first, cache = module.forward_cached(inputs[:, :4])
    second, updated = module.forward_cached(inputs[:, 4:], cache)

    torch.testing.assert_close(torch.cat((first, second), dim=1), full)
    assert updated[0].shape == (2, 2, 6, 4)
    assert updated[1].shape == updated[0].shape


def test_attention_rejects_context_overflow_and_invalid_dimensions() -> None:
    module = attention()

    with pytest.raises(ValueError, match="exceeds context"):
        module(torch.zeros(1, 9, 8))
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(10, 3, 8, 0.0)
