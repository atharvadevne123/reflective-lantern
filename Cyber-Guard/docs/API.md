# Cyber-Guard API Reference

All endpoints are versioned under `/api/v1`. Interactive OpenAPI docs are
served at `/docs`.

## Conventions

**Correlation IDs.** Send `X-Correlation-ID` and it is echoed on the response
and written to every log line for that request. If you omit it, one is
generated.

**Rate limiting.** Every endpoint except `/api/v1/health` is limited per
caller over a 60-second sliding window. Responses carry `X-RateLimit-Limit`
and `X-RateLimit-Remaining`; exceeding the limit returns `429` with
`Retry-After`.

**Timing.** Every response carries `X-Response-Time-Ms`.

## POST /api/v1/predict

Classify a connection into one of five threat classes.

| Field | Type | Constraint |
|-------|------|-----------|
| `src_bytes` | float | ≥ 0 |
| `dst_bytes` | float | ≥ 0 |
| `duration` | float | ≥ 0 |
| `protocol_type` | string | `tcp`, `udp`, `icmp` |
| `service` | string | `http`, `ftp`, `smtp`, `ssh`, `dns`, … |
| `flag` | string | `SF`, `S0`, `REJ`, `RSTO`, … |

Returns the predicted class, its confidence, the full probability
distribution, and an `anomaly` block. **When `anomaly.is_anomaly` is true,
treat the class label with low trust** — the connection sits outside the
training distribution, and the classifier is nonetheless obliged to pick one
of the five labels.

Unknown `protocol_type` returns `422`; negative byte counts return `422`.

## POST /api/v1/anomaly

Same request body. Scores outlier-ness without assigning a class, for triaging
traffic that matches none of the known categories.

`anomaly_score` is the sign-flipped IsolationForest decision score, so higher
means more anomalous.

## GET /api/v1/health

Reports `status` (`healthy` / `degraded`), `model_loaded`,
`anomaly_model_loaded`, `database_reachable`, and `version`. Exempt from rate
limiting. Suitable for both readiness and liveness probes.

## GET /api/v1/metrics

Query params: `hours` (default 24), `run_drift` (default false).

Returns prediction volume, per-class counts, and mean confidence over the
window, plus an inline drift result when `run_drift=true`.

## GET /api/v1/drift

Runs a two-sample KS test comparing the last 24 hours of `src_bytes` against
everything older than `REFERENCE_WINDOW_DAYS`.

Returns `ks_statistic`, `p_value`, and `drift_detected`. When either window
holds fewer than two rows the test is not computable and an `error` key is
returned instead — the endpoint still responds `200`, since "not enough data
yet" is a normal state for a fresh deployment, not a failure.
