from pathlib import Path

import pytest

from project_genesis.inference import BundleProvenance
from project_genesis.inference.load_probe import probe_passed, summarize_load_probe


def test_bundle_provenance_requires_dataset_sha256() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        BundleProvenance("abc123", "run-7", "not-a-digest")


def test_load_probe_summary_and_regression_gates() -> None:
    result = summarize_load_probe(
        [0.01, 0.02, 0.03, 0.04],
        failures=1,
        elapsed_seconds=0.5,
    )

    assert result.requests == 5
    assert result.successes == 4
    assert result.error_rate == pytest.approx(0.2)
    assert result.requests_per_second == pytest.approx(10)
    assert result.p50_milliseconds == pytest.approx(20)
    assert result.p95_milliseconds == pytest.approx(40)
    assert probe_passed(result, max_error_rate=0.2, max_p95_milliseconds=40)
    assert not probe_passed(result, max_error_rate=0.19)
    assert not probe_passed(result, max_error_rate=0.2, max_p95_milliseconds=39)


def test_release_and_operations_contracts_are_present() -> None:
    release = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    dependabot = Path(".github/dependabot.yml").read_text(encoding="utf-8")
    runbook = Path("docs/runbook.md").read_text(encoding="utf-8").lower()

    assert "actions/attest@v4" in release
    assert "gh release create" in release
    assert "--verify-tag" in release
    assert "attestations: write" in release
    assert "uv run --locked pytest -W error" in release
    assert 'package-ecosystem: "uv"' in dependabot
    assert "rollback" in runbook
    assert "bundle fingerprint" in runbook
