import torch
from torch import nn

from project_genesis.model import (
    GPTDecoder,
    ModelConfig,
    TransformerBlock,
    parameter_count,
)


def _config(*, tie_embeddings: bool = True) -> ModelConfig:
    return ModelConfig(
        vocab_size=31,
        context_length=8,
        d_model=12,
        n_heads=3,
        d_ff=36,
        dropout=0.0,
        bias=True,
        layer_norm_epsilon=1e-5,
        n_layers=2,
        initializer_range=0.02,
        tie_embeddings=tie_embeddings,
    )


def test_transformer_block_preserves_shape_and_backpropagates() -> None:
    block = TransformerBlock(_config())
    inputs = torch.randn(2, 5, 12, requires_grad=True)

    output = block(inputs)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_decoder_returns_logits_and_backpropagates() -> None:
    model = GPTDecoder(_config())
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    logits = model(token_ids)
    logits.mean().backward()

    assert logits.shape == (2, 3, 31)
    assert model.token_embedding.embedding.weight.grad is not None


def test_decoder_is_causal_end_to_end() -> None:
    model = GPTDecoder(_config()).eval()
    original = torch.tensor([[1, 2, 3, 4]])
    future_changed = torch.tensor([[1, 2, 3, 9]])

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(future_changed)

    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3])
    assert not torch.equal(original_logits[:, 3], changed_logits[:, 3])


def test_cached_decoder_matches_full_forward() -> None:
    model = GPTDecoder(_config()).eval()
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])

    full = model(token_ids)
    first, cache = model.forward_cached(token_ids[:, :3])
    second, updated = model.forward_cached(token_ids[:, 3:], cache)

    torch.testing.assert_close(torch.cat((first, second), dim=1), full)
    assert len(updated) == model.config.n_layers
    assert all(layer[0].shape[2] == 5 for layer in updated)


def test_decoder_rejects_empty_and_overlong_sequences() -> None:
    model = GPTDecoder(_config())

    for token_ids, message in (
        (torch.empty((1, 0), dtype=torch.long), "must not be empty"),
        (torch.zeros((1, 9), dtype=torch.long), "exceeds context length"),
    ):
        try:
            model(token_ids)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("expected invalid sequence to be rejected")


def test_embedding_weight_tying_is_configurable() -> None:
    tied = GPTDecoder(_config())
    untied = GPTDecoder(_config(tie_embeddings=False))

    assert tied.lm_head.weight is tied.token_embedding.embedding.weight
    assert untied.lm_head.weight is not untied.token_embedding.embedding.weight


def test_parameter_count_handles_tied_and_frozen_parameters() -> None:
    model = GPTDecoder(_config())
    expected = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count(model) == expected
    model.final_norm.weight.requires_grad_(False)
    assert parameter_count(model, trainable_only=True) == expected - 12


def test_initialization_uses_zero_biases_and_scaled_residual_outputs() -> None:
    torch.manual_seed(7)
    model = GPTDecoder(_config(tie_embeddings=False))
    linear_biases = [
        module.bias
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.bias is not None
    ]
    ordinary_std = model.blocks[0].attention.qkv_projection.weight.std().item()
    residual_std = model.blocks[0].attention.output_projection.weight.std().item()

    assert all(torch.count_nonzero(bias).item() == 0 for bias in linear_biases)
    assert 0.015 < ordinary_std < 0.025
    assert 0.007 < residual_std < 0.013
