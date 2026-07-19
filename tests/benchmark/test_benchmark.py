import math
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from project_genesis.benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkResult,
    CaseResult,
    CompletionCase,
    benchmark_fingerprint,
    compare_results,
    load_report,
    run_benchmark,
    run_completion_benchmark,
    save_report,
)
from project_genesis.evaluation import EvaluationResult, RegressionThresholds
from project_genesis.inference import FinishReason, GenerationConfig
from project_genesis.model import ModelConfig


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


class CompletionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = ModelConfig(5, 8, 4, 1, 8, 0.0, True, 1e-5, 1)
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: Tensor) -> Tensor:
        logits = torch.zeros(*inputs.shape, self.config.vocab_size)
        logits[..., 2] = 1
        return logits


def _evaluation(
    *,
    loss: float = 1.0,
    accuracy: float = 0.8,
    throughput: float = 100.0,
) -> EvaluationResult:
    return EvaluationResult(
        loss=loss,
        perplexity=math.exp(loss),
        token_accuracy=accuracy,
        tokens=100,
        batches=10,
        elapsed_seconds=1.0,
        tokens_per_second=throughput,
    )


def _benchmark() -> BenchmarkResult:
    definitions = (
        BenchmarkCase("language", "text", (0, 1), 2),
        BenchmarkCase("function", "code", (1, 2), 4),
    )
    cases = (
        CaseResult("language", "text", 2, 2, True),
        CaseResult("function", "code", 4, 3, False),
    )
    return BenchmarkResult(
        benchmark_fingerprint(definitions),
        cases,
        0.5,
        0.5,
        4.0,
    )


def test_named_benchmark_scores_cases_and_restores_mode() -> None:
    model = NextTokenModel()
    model.train()
    cases = (
        BenchmarkCase("language", "text", (0, 1), 2),
        BenchmarkCase("function", "code", (1, 2), 4),
    )

    result = run_benchmark(model, cases)

    assert result.accuracy == 0.5
    assert result.suite_fingerprint == benchmark_fingerprint(cases)
    assert [case.correct for case in result.cases] == [True, False]
    assert result.cases_per_second > 0
    assert model.training


def test_regression_gate_reports_every_failed_metric() -> None:
    thresholds = RegressionThresholds(0.03, 0.02, 0.9)

    assert compare_results(
        _evaluation(loss=1.02, accuracy=0.79, throughput=95),
        _evaluation(),
        thresholds,
    ).passed
    failed = compare_results(
        _evaluation(loss=1.1, accuracy=0.7, throughput=80),
        _evaluation(),
        thresholds,
    )
    assert not failed.passed
    assert len(failed.failures) == 3


def test_coding_completion_flow_scores_generated_token_sequences() -> None:
    config = GenerationConfig(1, 0.0, 0, 1.0, 1.0, (2,), False)
    cases = (
        CompletionCase("function-body", "code", (1,), (2,)),
        CompletionCase("wrong-answer", "code", (1,), (3,)),
    )

    result = run_completion_benchmark(
        CompletionModel(),  # type: ignore[arg-type]
        cases,
        config,
    )

    assert result.exact_match_accuracy == 0.5
    assert result.cases[0].finish_reason is FinishReason.STOP
    assert len(result.suite_fingerprint) == 64


def test_report_round_trip_is_canonical_and_atomic(tmp_path: Path) -> None:
    report = BenchmarkReport(
        _evaluation(),
        _benchmark(),
        {"model": "tiny", "dataset": "validation"},
    )
    path = tmp_path / "report.json"

    save_report(path, report)
    loaded = load_report(path)

    assert loaded == report
    assert loaded.fingerprint == report.fingerprint
    assert path.read_bytes().endswith(b"\n")


def test_report_rejects_unknown_structure(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"version":1}', encoding="utf-8")

    with pytest.raises(ValueError, match="fields are invalid"):
        load_report(path)
