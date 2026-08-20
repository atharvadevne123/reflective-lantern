# API Reference

Base URL: `http://localhost:8000`  ·  All endpoints are under `/api/v1`.

Interactive documentation is served at `/docs` (Swagger) and `/redoc`.

## Common headers

Every response includes:

| Header | Meaning |
|---|---|
| `X-Request-ID` | Correlation id; echoed from the request if supplied |
| `X-Response-Time-Ms` | Server-side handling time |
| `X-RateLimit-Limit` | Requests allowed per rolling minute |
| `X-RateLimit-Remaining` | Requests left in the current window |

## POST /api/v1/predict

Predicts delivery duration for one shipment.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "carrier": "DHL",
    "distance_km": 42.5,
    "weight_kg": 3.2,
    "route_type": "urban",
    "hour_of_day": 14,
    "day_of_week": 2
  }'
```

```json
{
  "predicted_minutes": 87.34,
  "predicted_hours": 1.456,
  "confidence": 0.9127,
  "model_version": "1.0.0",
  "request_id": "a3f9c1e2"
}
```

`confidence` is derived from the standard deviation across the ensemble's
sub-estimators, normalised against the prediction magnitude. Values near 1.0
mean XGBoost, LightGBM, and RandomForest closely agree.

### Errors

| Status | Cause |
|---|---|
| `422` | Unknown carrier or route type, or a numeric out of range |
| `429` | Rate limit exceeded; see `Retry-After` |
| `503` | Model not loaded yet |

## POST /api/v1/predict/batch

Scores 1-100 shipments in one round trip.

```json
{ "shipments": [ { "carrier": "DHL", "distance_km": 42.5, "weight_kg": 3.2,
                   "route_type": "urban", "hour_of_day": 14, "day_of_week": 2 } ] }
```

Returns `{ "predictions": [...], "count": 1 }`. Each element has the same
shape as a single `/predict` response. An invalid member rejects the whole
batch with `422`.

## GET /api/v1/health

```json
{ "status": "healthy", "model_version": "1.0.0", "model_loaded": true }
```

Returns `degraded` when the model failed to load. Suitable as a Kubernetes
readiness probe.

## GET /api/v1/metrics

Last computed 5-fold cross-validation metrics.

```json
{
  "rmse_mean": 30.1,
  "r2_mean": 0.995,
  "n_features": 13,
  "n_samples": 2000,
  "model_version": "1.0.0"
}
```

## GET /api/v1/drift

Runs a two-sample Kolmogorov–Smirnov test comparing the most recent
predictions against the reference window, per feature.

```json
{
  "status": "ok",
  "features": {
    "distance_km": { "ks_statistic": 0.081, "p_value": 0.412, "drift_detected": false },
    "weight_kg": { "ks_statistic": 0.203, "p_value": 0.008, "drift_detected": true }
  }
}
```

Drift is flagged at `p < 0.05` and written to the `drift_logs` table.
