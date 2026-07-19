# Production Runbook

This runbook defines application-level procedures. The operating environment
must separately define owners, escalation paths, replica count, resource limits,
backup retention, and measured recovery objectives.

## Deployment preflight

1. Record the image digest, release tag, bundle fingerprint, source revision,
   training-run ID, and dataset fingerprint.
2. Verify the release checksums and GitHub artifact attestation.
3. Keep the currently deployed immutable bundle available for rollback.
4. Mount the candidate bundle read-only and inject the API key through the
   platform's secret store.
5. Start one candidate replica. Require `/healthz` and `/readyz` to succeed.
6. Send a known generation request and confirm the returned bundle fingerprint.
7. Run a workload-specific probe before increasing traffic:

   ```console
   genesis-load-test --url https://candidate.example/v1/generate \
     --requests 100 --concurrency 4 --api-key "$GENESIS_API_KEY" \
     --max-error-rate 0.01 --max-p95-milliseconds 2000
   ```

   Thresholds are examples only. Establish them from representative hardware,
   bundles, prompts, and service objectives.
8. Shift traffic gradually using the deployment platform. Watch readiness,
   error rate, latency, memory, and accelerator utilization.

## Incident triage

1. Stop rollout changes and identify the release, image digest, and bundle
   fingerprint serving affected requests.
2. Use `X-Request-ID` to correlate client failures with structured service logs.
   Logs intentionally omit prompts, completions, authorization, and API keys.
3. Check `/healthz`. Failure means the process or network path is unavailable.
4. Check `/readyz`. A live but unready replica either failed bundle startup or
   encountered a fatal inference error. Remove it from traffic and restart it;
   do not repeatedly retry inference inside the failed process.
5. Check the platform for out-of-memory termination, disk pressure, throttling,
   driver errors, and upstream timeout or rate-limit events.
6. If the candidate differs from the last healthy release, roll back first and
   investigate offline.

## Rollback

1. Select the last known-good image digest and its exact immutable bundle
   fingerprint. Do not modify the failed bundle in place.
2. Deploy that image and mount that bundle read-only.
3. Confirm `/readyz` returns the expected bundle fingerprint.
4. Run the known generation request and the established load-probe gates.
5. Shift traffic back, then retain logs and artifact identities for the incident
   review.

If both candidate and rollback fail, verify shared infrastructure and secrets
before choosing an older release.

## Bundle recovery

Bundle storage must retain immutable copies outside the serving node. Restore a
bundle into a new directory, verify all manifest checksums by loading it in an
isolated candidate replica, and compare its fingerprint with the release record.
Never repair or overwrite a deployed bundle.

The operator must test restore procedures and set retention and recovery
objectives appropriate to the cost of recreating training artifacts.

## API key rotation

Create a new secret in the platform store, roll replicas so they receive it,
verify authenticated requests, then revoke the old value. Because the service
accepts one key, overlap without downtime requires upstream authentication or a
rolling traffic strategy. Never place keys in command history, images, bundles,
logs, or version control.

## Release verification

For a downloaded artifact:

```console
sha256sum --check SHA256SUMS
gh attestation verify project_genesis-0.1.0-py3-none-any.whl \
  --repo OWNER/REPOSITORY
```

Use the repository and release version that produced the artifact.
