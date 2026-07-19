# Compatibility Policy

## Package and inference bundles

Project Genesis uses semantic `major.minor.patch` versions. Before `1.0`, an
inference bundle is loadable only by a runtime with the same major and minor
version. Patch releases must retain bundle compatibility.

From `1.0` onward, a runtime may load bundles from the same major version when
the runtime version is not older than the bundle version. Major-version changes
may break compatibility. Prerelease and build suffixes do not change this core
version comparison.

The bundle schema has its own semantic version. Unknown schema versions and
unknown or missing manifest fields fail closed. A schema migration must be an
explicit offline operation that writes a new immutable bundle; loaders do not
silently rewrite artifacts.

Schema `2.0.0` introduces required provenance fields and intentionally rejects
Phase 10 schema `1.0.0` bundles. Re-export the source checkpoint and tokenizer
with explicit provenance rather than editing the old manifest.

Every bundle records:

- Project and bundle-schema versions.
- Model, weights, and tokenizer SHA-256 checksums.
- A canonical bundle fingerprint and tokenizer fingerprint.
- Source revision, training-run ID, and dataset SHA-256 fingerprint.

## Public HTTP API

The `/v1` prefix is the compatibility boundary. Patch releases may add optional
response fields but must not remove fields, change field meaning, or weaken
validation. Breaking request or response changes require `/v2`.

Health endpoints are operational contracts: `/healthz` reports process
liveness, while `/readyz` reports whether the runtime can receive inference
traffic.

## Configuration and checkpoints

YAML schemas reject unknown fields. Adding a required field is breaking unless a
safe default preserves existing behavior. Configuration migrations are
documented alongside the consuming subsystem.

Training checkpoints are internal resumability artifacts, not a stable public
exchange format. Resume with the same Project Genesis version unless a release
explicitly documents checkpoint compatibility.
