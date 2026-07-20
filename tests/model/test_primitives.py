import pytest
import torch
from torch.nn import functional as functional

from project_genesis.model import FeedForward, LayerNorm, RMSNorm, residual_add


def test_custom_layer_norm_matches_pytorch_and_backpropagates() -> None:
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    normalization = LayerNorm(8, 1e-5)

    actual = normalization(inputs)
    expected = functional.layer_norm(
        inputs,
        (8,),
        normalization.weight,
        normalization.bias,
        normalization.epsilon,
    )
    actual.sum().backward()

    torch.testing.assert_close(actual, expected)
    assert inputs.grad is not None
    assert normalization.weight.grad is not None


def test_custom_layer_norm_preserves_half_precision_output() -> None:
    normalization = LayerNorm(8, 1e-5)
    inputs = torch.randn(2, 3, 8, dtype=torch.float16)

    assert normalization(inputs).dtype is torch.float16


def test_feed_forward_preserves_shape_and_eval_is_deterministic() -> None:
    network = FeedForward(d_model=8, d_ff=32, dropout=0.5)
    inputs = torch.randn(2, 4, 8, requires_grad=True)
    network.eval()

    first = network(inputs)
    second = network(inputs)
    first.sum().backward()

    assert first.shape == inputs.shape
    torch.testing.assert_close(first, second)
    assert inputs.grad is not None


def test_rms_norm_and_swiglu_backpropagate() -> None:
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    normalization = RMSNorm(8, 1e-5)
    network = FeedForward(8, 24, 0.0, bias=False, activation="swiglu")

    output = network(normalization(inputs))
    output.mean().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None


def test_residual_add_rejects_broadcasting_and_preserves_gradients() -> None:
    inputs = torch.randn(2, 3, requires_grad=True)
    update = torch.randn(2, 3, requires_grad=True)

    residual_add(inputs, update).sum().backward()

    assert inputs.grad is not None
    assert update.grad is not None
    with pytest.raises(ValueError, match="identical shapes"):
        residual_add(torch.zeros(2, 3), torch.zeros(1, 3))
