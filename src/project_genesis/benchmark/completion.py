"""Exact-match autoregressive completion benchmarks."""

import hashlib
import json
from dataclasses import dataclass

import torch

from project_genesis.inference import (
    FinishReason,
    GenerationConfig,
    generate,
)
from project_genesis.model import GPTDecoder


@dataclass(frozen=True, slots=True)
class CompletionCase:
    """One named prompt and expected generated token suffix."""

    name: str
    category: str
    prompt_token_ids: tuple[int, ...]
    expected_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate identity and non-negative token sequences."""
        if not self.name.strip() or not self.category.strip():
            raise ValueError("completion name and category must be non-empty")
        if not self.prompt_token_ids or not self.expected_token_ids:
            raise ValueError("completion prompt and expected tokens must be non-empty")
        if any(token_id < 0 for token_id in (*self.prompt_token_ids, *self.expected_token_ids)):
            raise ValueError("completion token IDs cannot be negative")


@dataclass(frozen=True, slots=True)
class CompletionCaseResult:
    """Generated suffix and exact-match outcome for one case."""

    name: str
    category: str
    generated_token_ids: tuple[int, ...]
    expected_token_ids: tuple[int, ...]
    finish_reason: FinishReason
    exact_match: bool

    def __post_init__(self) -> None:
        """Validate stored completion consistency."""
        if not self.name.strip() or not self.category.strip():
            raise ValueError("completion result identity must be non-empty")
        if any(token_id < 0 for token_id in (*self.generated_token_ids, *self.expected_token_ids)):
            raise ValueError("completion result token IDs cannot be negative")
        if self.exact_match != (self.generated_token_ids == self.expected_token_ids):
            raise ValueError("completion exact_match does not match token sequences")


@dataclass(frozen=True, slots=True)
class CompletionBenchmarkResult:
    """Aggregate exact-match completion score."""

    suite_fingerprint: str
    cases: tuple[CompletionCaseResult, ...]
    exact_match_accuracy: float

    def __post_init__(self) -> None:
        """Validate suite identity and aggregate accuracy."""
        if len(self.suite_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.suite_fingerprint
        ):
            raise ValueError("suite_fingerprint must be lowercase SHA-256")
        if not self.cases:
            raise ValueError("completion result must contain cases")
        expected = sum(case.exact_match for case in self.cases) / len(self.cases)
        if self.exact_match_accuracy != expected:
            raise ValueError("completion accuracy does not match case results")


def run_completion_benchmark(
    model: GPTDecoder,
    cases: tuple[CompletionCase, ...],
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> CompletionBenchmarkResult:
    """Generate every case and report exact token-sequence accuracy."""
    if not cases:
        raise ValueError("at least one completion case is required")
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("completion case names must be unique")
    outcomes: list[CompletionCaseResult] = []
    for case in cases:
        generated = generate(
            model,
            case.prompt_token_ids,
            config,
            generator=generator,
        )
        outcomes.append(
            CompletionCaseResult(
                name=case.name,
                category=case.category,
                generated_token_ids=generated.generated_token_ids,
                expected_token_ids=case.expected_token_ids,
                finish_reason=generated.finish_reason,
                exact_match=generated.generated_token_ids == case.expected_token_ids,
            )
        )
    results = tuple(outcomes)
    return CompletionBenchmarkResult(
        suite_fingerprint=_completion_fingerprint(cases),
        cases=results,
        exact_match_accuracy=sum(result.exact_match for result in results) / len(results),
    )


def _completion_fingerprint(cases: tuple[CompletionCase, ...]) -> str:
    payload = [
        {
            "name": case.name,
            "category": case.category,
            "prompt_token_ids": case.prompt_token_ids,
            "expected_token_ids": case.expected_token_ids,
        }
        for case in cases
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
