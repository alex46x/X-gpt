"""Canonical, atomic benchmark reports."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from project_genesis.benchmark.core import (
    BenchmarkResult,
    CaseResult,
)
from project_genesis.evaluation import EvaluationResult
from project_genesis.utilities.files import atomic_write_text

REPORT_VERSION = 1


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Versioned evaluation, benchmark, and experiment metadata."""

    evaluation: EvaluationResult
    benchmark: BenchmarkResult
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze sorted, non-empty string metadata."""
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.metadata.items()
        ):
            raise ValueError("report metadata must contain non-empty strings")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(sorted(self.metadata.items()))),
        )

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the canonical report."""
        return hashlib.sha256(report_json(self).encode()).hexdigest()


def report_json(report: BenchmarkReport) -> str:
    """Serialize a report as canonical strict JSON."""
    evaluation = report.evaluation
    benchmark = report.benchmark
    payload = {
        "version": REPORT_VERSION,
        "metadata": dict(report.metadata),
        "evaluation": {
            "loss": evaluation.loss,
            "perplexity": evaluation.perplexity,
            "token_accuracy": evaluation.token_accuracy,
            "tokens": evaluation.tokens,
            "batches": evaluation.batches,
            "elapsed_seconds": evaluation.elapsed_seconds,
            "tokens_per_second": evaluation.tokens_per_second,
        },
        "benchmark": {
            "suite_fingerprint": benchmark.suite_fingerprint,
            "accuracy": benchmark.accuracy,
            "elapsed_seconds": benchmark.elapsed_seconds,
            "cases_per_second": benchmark.cases_per_second,
            "cases": [
                {
                    "name": case.name,
                    "category": case.category,
                    "expected_token_id": case.expected_token_id,
                    "predicted_token_id": case.predicted_token_id,
                    "correct": case.correct,
                }
                for case in benchmark.cases
            ],
        },
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def save_report(path: Path, report: BenchmarkReport) -> None:
    """Atomically save a canonical benchmark report."""
    atomic_write_text(path, f"{report_json(report)}\n")


def load_report(path: Path) -> BenchmarkReport:
    """Load and strictly validate a Project Genesis benchmark report."""
    source = path.expanduser().resolve()
    try:
        loaded: object = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to load benchmark report {path}: {error}") from error
    root = _mapping(loaded, "report")
    _require_keys(root, {"version", "metadata", "evaluation", "benchmark"}, "report")
    if root["version"] != REPORT_VERSION:
        raise ValueError("unsupported benchmark report version")

    metadata_values = _mapping(root["metadata"], "metadata")
    metadata = {key: _string(value, f"metadata.{key}") for key, value in metadata_values.items()}
    evaluation = _parse_evaluation(_mapping(root["evaluation"], "evaluation"))
    benchmark = _parse_benchmark(_mapping(root["benchmark"], "benchmark"))
    return BenchmarkReport(evaluation, benchmark, metadata)


def _parse_evaluation(values: dict[str, object]) -> EvaluationResult:
    fields = {
        "loss",
        "perplexity",
        "token_accuracy",
        "tokens",
        "batches",
        "elapsed_seconds",
        "tokens_per_second",
    }
    _require_keys(values, fields, "evaluation")
    return EvaluationResult(
        loss=_number(values["loss"], "evaluation.loss"),
        perplexity=_number(values["perplexity"], "evaluation.perplexity"),
        token_accuracy=_number(
            values["token_accuracy"],
            "evaluation.token_accuracy",
        ),
        tokens=_integer(values["tokens"], "evaluation.tokens"),
        batches=_integer(values["batches"], "evaluation.batches"),
        elapsed_seconds=_number(
            values["elapsed_seconds"],
            "evaluation.elapsed_seconds",
        ),
        tokens_per_second=_number(
            values["tokens_per_second"],
            "evaluation.tokens_per_second",
        ),
    )


def _parse_benchmark(values: dict[str, object]) -> BenchmarkResult:
    _require_keys(
        values,
        {
            "suite_fingerprint",
            "accuracy",
            "elapsed_seconds",
            "cases_per_second",
            "cases",
        },
        "benchmark",
    )
    case_values = values["cases"]
    if not isinstance(case_values, list):
        raise ValueError("benchmark.cases must be a list")
    cases: list[CaseResult] = []
    for index, value in enumerate(case_values):
        location = f"benchmark.cases[{index}]"
        case = _mapping(value, location)
        _require_keys(
            case,
            {
                "name",
                "category",
                "expected_token_id",
                "predicted_token_id",
                "correct",
            },
            location,
        )
        cases.append(
            CaseResult(
                name=_string(case["name"], f"{location}.name"),
                category=_string(case["category"], f"{location}.category"),
                expected_token_id=_integer(
                    case["expected_token_id"],
                    f"{location}.expected_token_id",
                ),
                predicted_token_id=_integer(
                    case["predicted_token_id"],
                    f"{location}.predicted_token_id",
                ),
                correct=_boolean(case["correct"], f"{location}.correct"),
            )
        )
    return BenchmarkResult(
        suite_fingerprint=_string(
            values["suite_fingerprint"],
            "benchmark.suite_fingerprint",
        ),
        cases=tuple(cases),
        accuracy=_number(values["accuracy"], "benchmark.accuracy"),
        elapsed_seconds=_number(
            values["elapsed_seconds"],
            "benchmark.elapsed_seconds",
        ),
        cases_per_second=_number(
            values["cases_per_second"],
            "benchmark.cases_per_second",
        ),
    )


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return value


def _require_keys(
    values: Mapping[str, object],
    required: set[str],
    location: str,
) -> None:
    if set(values) != required:
        raise ValueError(f"{location} fields are invalid")


def _string(value: object, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string")
    return value


def _number(value: object, location: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{location} must be a number")
    return float(value)


def _integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{location} must be an integer")
    return value


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be a boolean")
    return value
