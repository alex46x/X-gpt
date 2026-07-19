# Phase 10 — Deployment

## Phase overview

Phase 10 packages trained inference artifacts into a verified immutable bundle
and exposes generation and chat through one bounded HTTP service. External
container artifacts remain in `deployment/`; reusable bundle and transport code
stays beside inference.

## Architecture

```text
GPTDecoder + ByteBPETokenizer
  → atomic inference bundle
      ├── model.yaml
      ├── model.pt
      ├── tokenizer.json
      └── manifest.json

verified bundle + inference YAML
  → ServiceRuntime
  → FastAPI
      ├── GET  /healthz
      ├── GET  /readyz
      ├── POST /v1/generate
      └── POST /v1/chat
  → Uvicorn
  → non-root container
```

The service is stateless. Conversation history is supplied in each chat request,
so replicas require no session database or affinity.

## Inference bundles

`save_bundle` requires equal model and tokenizer vocabulary sizes and refuses to
overwrite an existing destination. It writes into a sibling temporary directory,
flushes model weights, writes the existing validated tokenizer format, records
the package version and SHA-256 checksums, then renames the complete directory
into place.

`load_bundle` verifies:

- Exact manifest fields and schema version.
- Canonical manifest fingerprint.
- Model configuration, weight, and tokenizer file checksums.
- Strict model YAML through the existing loader.
- Tokenizer serialization and internal fingerprint.
- Model/tokenizer vocabulary compatibility.
- PyTorch weights through restricted weights-only loading.

The loaded model is placed on the configured device and switched to evaluation
mode. Training optimizer, scheduler, scaler, and RNG state are excluded.

## HTTP contracts

FastAPI and Pydantic provide strict request schemas and reject unknown fields.
Generation requests contain a prompt and optional bounded sampling overrides.
Chat requests contain complete role-tagged history and must end with a user
message.

Responses contain decoded text, exact generated token IDs, finish reason, and
the serving bundle fingerprint. This makes every response attributable to one
artifact version.

The console entry point is:

```console
genesis-serve
```

## Health and observability

`/healthz` reports process liveness without requiring model access. `/readyz`
returns success only after bundle loading and includes the bundle fingerprint.

Every request receives an `X-Request-ID`. The service emits compact JSON records
with request ID, method, path, status, and duration. Prompts, generated content,
authorization headers, and API keys are never logged.

## Security controls

- Optional bearer authentication with constant-time comparison on `/v1/*`.
- Configurable request-body, prompt-character, and generated-token limits.
- Strict JSON fields and generation bounds.
- No permissive CORS middleware.
- Proxy headers disabled by default.
- Non-root container user.
- Read-only bundle mounting and read-only container operation documented.
- No generated-code execution.

Set `GENESIS_API_KEY` unless an authenticated ingress enforces equivalent policy.
TLS, network policy, distributed rate limiting, and secret injection belong to
the deployment platform rather than this process.

## Concurrency and scaling

One process-local lock serializes access to one model instance because generation
changes model mode and accelerator kernels share device state. Scale with process
or container replicas. Continuous batching should be introduced only after
measurements show the lock is the limiting factor.

## Container and CI

`deployment/Dockerfile` uses Python 3.13 slim, the committed uv lock, a non-root
UID, a standard-library health check, and the `genesis-serve` entry point. Model
artifacts are mounted rather than copied into the image.

The image is CPU-only because the committed torch source is CPU-only. CUDA needs
a separately locked image and accelerator CI. CI builds the deployment image
after both Python quality jobs pass.

Docker is unavailable in the current local environment, so local verification
covers the Dockerfile contents and the CI job owns the executable image build.

## Testing

Tests cover atomic bundle round-trip, parameter equality, immutable destination
policy, checksum tampering, health and readiness, bearer authentication,
generation, chat, request IDs, prompt limits, and generation limits.

## Acceptance commands

```powershell
uv sync --locked --group dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest -W error
uv build
```

CI additionally runs:

```console
docker build -f deployment/Dockerfile -t project-genesis:ci .
```
