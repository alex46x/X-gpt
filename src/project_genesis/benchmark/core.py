"""Named next-token benchmarks and evaluation regression gates."""

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from project_genesis.evaluation import EvaluationResult, RegressionThresholds


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One named next-token prediction case."""

    name: str
    category: str
    prompt_token_ids: tuple[int, ...]
    expected_token_id: int

    def __post_init__(self) -> None:
        """Validate case identity and token IDs."""
        if not self.name.strip() or not self.category.strip():
            raise ValueError("benchmark name and category must be non-empty")
        if not self.prompt_token_ids:
            raise ValueError("benchmark prompt must not be empty")
        if self.expected_token_id < 0 or any(token_id < 0 for token_id in self.prompt_token_ids):
            raise ValueError("benchmark token IDs cannot be negative")


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Prediction outcome for one benchmark case."""

    name: str
    category: str
    expected_token_id: int
    predicted_token_id: int
    correct: bool

    def __post_init__(self) -> None:
        """Validate stored prediction consistency."""
        if not self.name.strip() or not self.category.strip():
            raise ValueError("case result name and category must be non-empty")
        if self.expected_token_id < 0 or self.predicted_token_id < 0:
            raise ValueError("case result token IDs cannot be negative")
        if self.correct != (self.expected_token_id == self.predicted_token_id):
            raise ValueError("case result correctness does not match token IDs")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate named-case accuracy and timing."""

    suite_fingerprint: str
    cases: tuple[CaseResult, ...]
    accuracy: float
    elapsed_seconds: float
    cases_per_second: float

    def __post_init__(self) -> None:
        """Validate aggregate consistency."""
        if len(self.suite_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.suite_fingerprint
        ):
            raise ValueError("suite_fingerprint must be lowercase SHA-256")
        if not self.cases:
            raise ValueError("benchmark result must contain cases")
        expected_accuracy = sum(case.correct for case in self.cases) / len(self.cases)
        if self.accuracy != expected_accuracy:
            raise ValueError("benchmark accuracy does not match case results")
        if (
            not math.isfinite(self.elapsed_seconds)
            or not math.isfinite(self.cases_per_second)
            or self.elapsed_seconds <= 0
            or self.cases_per_second <= 0
        ):
            raise ValueError("benchmark timing must be positive")


@dataclass(frozen=True, slots=True)
class RegressionResult:
    """Outcome of comparing current evaluation metrics to a baseline."""

    passed: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require the pass flag to agree with failures."""
        if self.passed != (not self.failures):
            raise ValueError("passed must be true exactly when failures are empty")


def run_benchmark(
    model: nn.Module,
    cases: tuple[BenchmarkCase, ...],
) -> BenchmarkResult:
    """Run deterministic next-token cases without changing model mode."""
    if not cases:
        raise ValueError("at least one benchmark case is required")
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("benchmark case names must be unique")
    parameter = next(model.parameters(), None)
    device = torch.device("cpu") if parameter is None else parameter.device
    was_training = model.training
    model.eval()
    results: list[CaseResult] = []
    _synchronize(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            # ponytail: one forward per case; bucket by length when suite runtime matters.
            for case in cases:
                inputs = torch.tensor(
                    [case.prompt_token_ids],
                    dtype=torch.long,
                    device=device,
                )
                logits = cast(Tensor, model(inputs))
                predicted = int(logits[0, -1].argmax().item())
                results.append(
                    CaseResult(
                        name=case.name,
                        category=case.category,
                        expected_token_id=case.expected_token_id,
                        predicted_token_id=predicted,
                        correct=predicted == case.expected_token_id,
                    )
                )
        _synchronize(device)
    finally:
        model.train(was_training)
    elapsed = time.perf_counter() - started
    outcomes = tuple(results)
    return BenchmarkResult(
        suite_fingerprint=benchmark_fingerprint(cases),
        cases=outcomes,
        accuracy=sum(case.correct for case in outcomes) / len(outcomes),
        elapsed_seconds=elapsed,
        cases_per_second=len(outcomes) / elapsed,
    )


def benchmark_fingerprint(cases: tuple[BenchmarkCase, ...]) -> str:
    """Return the canonical SHA-256 identity of an ordered benchmark suite."""
    payload = [
        {
            "name": case.name,
            "category": case.category,
            "prompt_token_ids": case.prompt_token_ids,
            "expected_token_id": case.expected_token_id,
        }
        for case in cases
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compare_results(
    current: EvaluationResult,
    baseline: EvaluationResult,
    thresholds: RegressionThresholds,
) -> RegressionResult:
    """Compare current quality and throughput with an accepted baseline."""
    failures: list[str] = []
    if current.loss > baseline.loss + thresholds.max_loss_increase:
        failures.append("loss increased beyond threshold")
    if current.token_accuracy < baseline.token_accuracy - thresholds.max_accuracy_drop:
        failures.append("token accuracy dropped beyond threshold")
    if current.tokens_per_second < (baseline.tokens_per_second * thresholds.min_throughput_ratio):
        failures.append("throughput dropped below threshold")
    return RegressionResult(passed=not failures, failures=tuple(failures))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
