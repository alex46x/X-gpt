"""Dependency-free concurrent HTTP load probe."""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class LoadProbeResult:
    """Aggregate request success, throughput, and latency measurements."""

    requests: int
    successes: int
    failures: int
    elapsed_seconds: float
    requests_per_second: float
    p50_milliseconds: float
    p95_milliseconds: float
    max_milliseconds: float

    def __post_init__(self) -> None:
        """Validate counts and finite non-negative measurements."""
        if self.requests <= 0 or self.successes + self.failures != self.requests:
            raise ValueError("load probe counts are inconsistent")
        measurements = (
            self.elapsed_seconds,
            self.requests_per_second,
            self.p50_milliseconds,
            self.p95_milliseconds,
            self.max_milliseconds,
        )
        if not all(math.isfinite(value) and value >= 0 for value in measurements):
            raise ValueError("load probe measurements must be finite and non-negative")
        if self.elapsed_seconds <= 0 or self.requests_per_second <= 0:
            raise ValueError("load probe elapsed time and throughput must be positive")

    @property
    def error_rate(self) -> float:
        """Return failed requests divided by attempted requests."""
        return self.failures / self.requests


def summarize_load_probe(
    latencies_seconds: list[float],
    *,
    failures: int,
    elapsed_seconds: float,
) -> LoadProbeResult:
    """Build deterministic aggregate metrics from successful request latencies."""
    if failures < 0 or elapsed_seconds <= 0:
        raise ValueError("failures cannot be negative and elapsed time must be positive")
    if any(not math.isfinite(value) or value < 0 for value in latencies_seconds):
        raise ValueError("latencies must be finite and non-negative")
    ordered = sorted(latencies_seconds)
    requests = len(ordered) + failures
    if not requests:
        raise ValueError("at least one request result is required")
    return LoadProbeResult(
        requests=requests,
        successes=len(ordered),
        failures=failures,
        elapsed_seconds=elapsed_seconds,
        requests_per_second=requests / elapsed_seconds,
        p50_milliseconds=_percentile(ordered, 0.50) * 1000,
        p95_milliseconds=_percentile(ordered, 0.95) * 1000,
        max_milliseconds=(ordered[-1] * 1000 if ordered else 0.0),
    )


def probe_passed(
    result: LoadProbeResult,
    *,
    max_error_rate: float,
    max_p95_milliseconds: float | None = None,
) -> bool:
    """Return whether configured error-rate and optional latency gates pass."""
    if not 0 <= max_error_rate <= 1:
        raise ValueError("max_error_rate must be in [0, 1]")
    if max_p95_milliseconds is not None and max_p95_milliseconds <= 0:
        raise ValueError("max_p95_milliseconds must be positive")
    return result.error_rate <= max_error_rate and (
        max_p95_milliseconds is None or result.p95_milliseconds <= max_p95_milliseconds
    )


def run_load_probe(
    url: str,
    prompt: str,
    *,
    requests: int,
    concurrency: int,
    timeout_seconds: float,
    api_key: str | None = None,
) -> LoadProbeResult:
    """Send concurrent generation requests and summarize successful latencies."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must use http or https")
    if not prompt or requests <= 0 or concurrency <= 0 or timeout_seconds <= 0:
        raise ValueError("prompt and positive probe bounds are required")
    payload = json.dumps(
        {"prompt": prompt, "max_new_tokens": 1, "temperature": 0},
        separators=(",", ":"),
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    def send() -> float:
        started = time.perf_counter()
        request = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=timeout_seconds) as response:
            response.read()
            if not 200 <= response.status < 300:
                raise RuntimeError(f"HTTP {response.status}")
        return time.perf_counter() - started

    latencies: list[float] = []
    failures = 0
    started = time.perf_counter()
    # ponytail: ceiling — bounded threads are enough for this deployment smoke test.
    with ThreadPoolExecutor(max_workers=min(concurrency, requests)) as executor:
        futures = [executor.submit(send) for _ in range(requests)]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except (OSError, RuntimeError):
                failures += 1
    return summarize_load_probe(
        latencies,
        failures=failures,
        elapsed_seconds=time.perf_counter() - started,
    )


def _percentile(ordered: list[float], percentile: float) -> float:
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def main() -> None:
    """Run the load probe and fail the process when thresholds regress."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/generate")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--api-key")
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--max-p95-milliseconds", type=float)
    arguments = parser.parse_args()
    result = run_load_probe(
        arguments.url,
        arguments.prompt,
        requests=arguments.requests,
        concurrency=arguments.concurrency,
        timeout_seconds=arguments.timeout_seconds,
        api_key=arguments.api_key,
    )
    print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
    if not probe_passed(
        result,
        max_error_rate=arguments.max_error_rate,
        max_p95_milliseconds=arguments.max_p95_milliseconds,
    ):
        raise SystemExit(1)
