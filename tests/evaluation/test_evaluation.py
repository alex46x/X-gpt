import math
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from project_genesis.evaluation import (
    EvaluationResult,
    evaluate_model,
    load_evaluation_config,
)


class NextTokenModel(nn.Module):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: Tensor) -> Tensor:
        predictions = (inputs + 1) % self.vocab_size
        logits = torch.zeros(
            *inputs.shape,
            self.vocab_size,
            device=inputs.device,
        )
        return logits.scatter(-1, predictions.unsqueeze(-1), 10.0)


def test_default_evaluation_config_loads_with_override() -> None:
    config = load_evaluation_config(
        Path("configs/evaluation/default.yaml"),
        ["evaluation.max_batches=2"],
    )

    assert config.max_batches == 2
    assert config.regression.min_throughput_ratio == 0.9


def test_evaluation_is_token_weighted_and_restores_training_mode() -> None:
    model = NextTokenModel()
    model.train()
    batches = [
        (torch.tensor([[0, 1]]), torch.tensor([[1, 2]])),
        (
            torch.tensor([[2, 3, 4, 0]]),
            torch.tensor([[3, 4, 0, 0]]),
        ),
    ]

    result = evaluate_model(model, batches)

    assert isinstance(result, EvaluationResult)
    assert result.tokens == 6
    assert result.batches == 2
    assert result.token_accuracy == 5 / 6
    assert result.perplexity == pytest.approx(math.exp(result.loss))
    assert result.tokens_per_second > 0
    assert model.training


def test_evaluation_honors_batch_limit_and_rejects_empty_input() -> None:
    model = NextTokenModel()
    batch = (torch.tensor([[0, 1]]), torch.tensor([[1, 2]]))

    assert evaluate_model(model, [batch, batch], max_batches=1).batches == 1
    with pytest.raises(ValueError, match="at least one"):
        evaluate_model(model, [])
