# Traffic-Pulse API Documentation

Base URL: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs` (Swagger UI)

## Endpoints

### `GET /health`

Liveness probe.

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `ok` when the app is alive |
| `model_version` | string | Semantic version of the loaded model |
| `models_loaded` | string[] | Ensemble member names (`xgb`, `lgbm`) |

### `GET /api/v1/model-info`

Model metadata — version, ensemble members, the 26 feature names, and
congestion label mapping.

### `POST /api/v1/predict`

Single-route congestion prediction.

Required fields: `route_id`, `hour` (0-23), `day_of_week` (0-6),
`vehicle_count` (0-10000), `avg_speed_kmh` (0-200).

Optional fields: `month`, `road_type` (highway | arterial | collector |
local | expressway), `incident_count`, `temperature_celsius`, `is_raining`,
plus lag/rolling overrides (`lag_1h`, `lag_2h`, `lag_4h`, `rolling_mean_6h`,
`rolling_std_6h`, `rolling_mean_24h`).

Response: congestion level 0-3 with label, class probabilities, incident
score, and model version. Every prediction is persisted for monitoring.

### `POST /api/v1/predict/batch`

Same schema as `/predict` but accepts a JSON array (max 100 items) and
returns an array of predictions.

### `POST /api/v1/drift`

Two-sample Kolmogorov-Smirnov drift test.

Request: `feature_name` (string), `reference` (>=10 floats), `current`
(>=10 floats).

Response: `ks_statistic`, `p_value`, `drift_detected` (p < 0.05). Results
are persisted to `drift_logs` and gate the automated retraining pipeline.

### `GET /api/v1/metrics`

Monitoring snapshot: prediction volume, congestion level distribution,
active drift alerts, and training metrics (per-model CV AUC).

## Headers

Every response carries:

- `X-Correlation-ID` — echoed from the request or generated server-side.
- `X-Response-Time-Ms` — server-side processing time.

Rate limiting: 120 requests/minute per client IP (HTTP 429 beyond that,
with a `Retry-After: 60` header).
