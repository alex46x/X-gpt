"""Reproducible next-token benchmarks, reports, and regression gates."""

from project_genesis.benchmark.completion import (
    CompletionBenchmarkResult,
    CompletionCase,
    CompletionCaseResult,
    run_completion_benchmark,
)
from project_genesis.benchmark.core import (
    BenchmarkCase,
    BenchmarkResult,
    CaseResult,
    RegressionResult,
    benchmark_fingerprint,
    compare_results,
    run_benchmark,
)
from project_genesis.benchmark.report import (
    BenchmarkReport,
    load_report,
    report_json,
    save_report,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkReport",
    "BenchmarkResult",
    "CaseResult",
    "CompletionBenchmarkResult",
    "CompletionCase",
    "CompletionCaseResult",
    "RegressionResult",
    "benchmark_fingerprint",
    "compare_results",
    "load_report",
    "report_json",
    "run_benchmark",
    "run_completion_benchmark",
    "save_report",
]
