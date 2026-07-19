# Deployment

The foundational image serves one CPU inference-bundle replica. Build from the
repository root:

```console
docker build -f deployment/Dockerfile -t project-genesis:0.1.0 .
```

Run with a read-only bundle mount:

```console
docker run --rm -p 8000:8000 \
  --read-only \
  --tmpfs /tmp \
  --mount type=bind,src=/absolute/path/to/bundle,dst=/models/genesis,readonly \
  -e GENESIS_BUNDLE=/models/genesis \
  -e GENESIS_API_KEY=replace-with-a-secret \
  project-genesis:0.1.0
```

## Environment

- `GENESIS_BUNDLE` is required and identifies a verified bundle directory.
- `GENESIS_INFERENCE_CONFIG` defaults to
  `configs/inference/default.yaml`.
- `GENESIS_DEVICE` defaults to `cpu`.
- `GENESIS_API_KEY` enables bearer authentication for `/v1/*`.
- `GENESIS_MAX_REQUEST_BYTES` defaults to `262144`.
- `GENESIS_MAX_PROMPT_CHARACTERS` defaults to `32768`.
- `GENESIS_MAX_NEW_TOKENS` defaults to `512`.
- `GENESIS_HOST` and `GENESIS_PORT` default to `0.0.0.0:8000`.
- `GENESIS_LOG_LEVEL` defaults to `INFO`.

`GET /healthz` is a process liveness check. `GET /readyz` succeeds only after a
bundle is loaded and returns its fingerprint. `POST /v1/generate` and
`POST /v1/chat` are stateless JSON APIs documented by the service OpenAPI schema.

## Security and scaling

The container runs as UID `10001`, does not contain model artifacts, and can use
a read-only root filesystem. TLS, network policy, rate limiting, request-body
limits at the byte level, and secret injection belong at the ingress/runtime
boundary. Set `GENESIS_API_KEY` unless equivalent upstream authentication is
enforced.

One process serializes access to one model instance. Scale with process or
container replicas. Add continuous batching only after production measurements
show the process-local lock is the throughput bottleneck.

The image is CPU-only because the committed PyTorch lock is CPU-only. A CUDA
image requires a separately locked accelerator dependency set and GPU CI.
