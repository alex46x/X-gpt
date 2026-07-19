from pathlib import Path

import pytest
import torch

from project_genesis.inference import (
    FinishReason,
    GenerationConfig,
    generate,
    load_generation_config,
    sample_next_token,
)
from project_genesis.model import GPTDecoder, ModelConfig


def _model() -> GPTDecoder:
    return GPTDecoder(
        ModelConfig(
            vocab_size=16,
            context_length=8,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.0,
            bias=True,
            layer_norm_epsilon=1e-5,
            n_layers=2,
        )
    )


def _config(
    *,
    temperature: float = 0.0,
    stop_token_ids: tuple[int, ...] = (),
    use_cache: bool = True,
    max_new_tokens: int = 3,
    repetition_penalty: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_token_ids,
        use_cache=use_cache,
    )


def test_default_generation_config_loads_with_override() -> None:
    config = load_generation_config(
        Path("configs/inference/default.yaml"),
        ["inference.temperature=0.0", "inference.use_cache=false"],
    )

    assert config.temperature == 0
    assert not config.use_cache
    assert config.stop_token_ids == (2,)


def test_cached_and_uncached_greedy_generation_match_and_restore_mode() -> None:
    model = _model()
    model.train()
    prompt = (1, 2, 3)

    cached = generate(model, prompt, _config())
    uncached = generate(model, prompt, _config(use_cache=False))

    assert cached == uncached
    assert len(cached.generated_token_ids) == 3
    assert cached.finish_reason is FinishReason.LENGTH
    assert model.training


def test_generation_stops_and_reports_context_exhaustion() -> None:
    model = _model()
    for parameter in model.parameters():
        parameter.data.zero_()

    stopped = generate(model, (1,), _config(stop_token_ids=(0,)))
    exhausted = generate(
        model,
        (1, 2, 3, 4, 5, 6, 7),
        _config(max_new_tokens=3),
    )

    assert stopped.generated_token_ids == (0,)
    assert stopped.finish_reason is FinishReason.STOP
    assert len(exhausted.generated_token_ids) == 1
    assert exhausted.finish_reason is FinishReason.CONTEXT


def test_sampling_applies_repetition_and_seeded_probability_filters() -> None:
    repeated = sample_next_token(
        torch.tensor([2.0, 1.5]),
        (0,),
        _config(repetition_penalty=2.0),
    )
    first_generator = torch.Generator().manual_seed(9)
    second_generator = torch.Generator().manual_seed(9)
    sampling = _config(temperature=1.0, top_k=2, top_p=0.9)

    first = sample_next_token(
        torch.tensor([0.0, 1.0, 2.0]),
        (),
        sampling,
        generator=first_generator,
    )
    second = sample_next_token(
        torch.tensor([0.0, 1.0, 2.0]),
        (),
        sampling,
        generator=second_generator,
    )

    assert repeated == 1
    assert first == second
    assert first in {1, 2}


def test_generation_rejects_invalid_prompt_tokens() -> None:
    with pytest.raises(ValueError, match="outside"):
        generate(_model(), (16,), _config())
