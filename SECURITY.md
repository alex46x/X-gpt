# Security Policy

## Supported versions

Project Genesis is pre-1.0. Only the latest released `0.1.x` version receives
security fixes. Upgrade to the newest patch before reporting a problem.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's
private vulnerability reporting feature. If private reporting is unavailable,
contact the repository owner through a privately published maintainer channel.

Include the affected version, reproduction steps, impact, and any known
mitigation. Do not include real API keys, model data, prompts, or other secrets.

The maintainers will confirm receipt, investigate scope, coordinate a fix, and
publish an advisory when disclosure is safe. Response deadlines are not promised
until the project has an explicitly staffed operational support policy.

## Security boundary

Inference bundles are authenticated by checksums and provenance metadata, not
encrypted. API bearer authentication is optional and must be combined with TLS.
Generated code is untrusted output and this repository never executes it.
Ingress controls, secret storage, network policy, artifact access control, and
host isolation remain deployment-platform responsibilities.
