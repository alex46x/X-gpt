# Phase 11 - Production Hardening

## Phase overview

Phase 11 makes the existing bundle and service boundaries diagnosable,
compatible, recoverable, and releasable. It adds no orchestration platform or
runtime dependency.

## Architecture

```text
training identity + dataset fingerprint + source revision
  -> immutable verified inference bundle
  -> semantic runtime compatibility check
  -> serving runtime
       |-- healthy and ready
       `-- fatal inference failure -> live but unready

tag matching pyproject version
  -> locked quality checks
  -> source and wheel builds + SHA-256 list
  -> GitHub artifact attestation
  -> immutable GitHub release
```

The dependency-free load probe targets the existing HTTP boundary and reports
request count, failures, throughput, p50, p95, and maximum latency. It exits
nonzero when explicit error or latency gates fail.

## Design decisions

- Bundle provenance is required, not inferred from a mutable working directory.
- Dataset identity is a lowercase SHA-256 digest. Source and run identifiers
  remain deployment-specific non-empty strings.
- Pre-1.0 compatibility is same-major/same-minor. After 1.0, runtimes can load
  non-newer bundles from the same major version.
- Bundle corruption, version mismatch, and unknown fields fail before serving.
- Runtime/device failures fail the replica's readiness and return a generic 503.
  The process stays live so the platform can collect diagnostics and restart it.
- The load probe uses the standard library because it needs only bounded HTTP
  concurrency and percentile gates.
- Release tags must exactly match the package version. Releases include source,
  wheel, checksums, and GitHub-hosted build provenance.
- PyPI and container-registry publication are deferred until repository
  ownership, credentials, and publication targets are explicitly selected.
- Kubernetes, cloud, Prometheus, and alerting manifests remain operator-owned
  because no target environment or measured objectives exist.

## Failure and recovery model

The service distinguishes liveness from readiness. Invalid client input remains
a bounded 4xx response. A runtime or floating-point failure is treated as unsafe
model state: the active request receives a non-sensitive 503, subsequent
inference is rejected, and readiness fails. The deployment platform replaces
the replica.

Rollback selects a previously verified image digest and bundle fingerprint.
Artifacts are never repaired or overwritten in place.

## Supply-chain policy

The release workflow runs locked formatting, linting, typing, and tests before
building. It publishes SHA-256 checksums and uses GitHub's artifact attestation
action for provenance. Dependabot opens bounded weekly updates for uv,
GitHub Actions, and Docker dependencies; CI remains the acceptance gate.

## Testing

Tests cover provenance validation, semantic compatibility, fatal runtime failure
isolation, readiness behavior, deterministic load metrics, regression gates,
and the presence of release and rollback contracts. Existing bundle, service,
model, training, inference, and data tests continue as regression coverage.

The container build remains executable in CI because Docker is unavailable in
the local development environment.

## Acceptance commands

```powershell
uv sync --locked --group dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest -W error
uv build
```

No automatic Phase 12 is defined. Further production work must begin with a
real deployment target and measured reliability, quality, throughput, and cost
requirements.
