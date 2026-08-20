# Logistics-Flow

![CI](https://github.com/atharvadevne123/Logistics-Flow/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Last-mile delivery time prediction and logistics optimization API using an
XGBoost + LightGBM + RandomForest ensemble, with route risk scoring, carrier
performance analytics, KS-test drift monitoring, and Airflow retraining.

![Architecture](screenshots/architecture.png)

---

## Overview

Logistics-Flow estimates how long a parcel will take to reach its destination,
given carrier, distance, weight, route type, and dispatch time. It is built as
a production service rather than a notebook: every prediction is validated,
logged, and monitored for distribution drift, and the model retrains itself on
a weekly Airflow schedule.

**What it does**

- Predicts delivery duration in minutes with an ensemble confidence score
- Scores carrier-specific delay risk (DHL, FedEx, UPS, USPS, Amazon)
- Engineers 13 features including cyclical time encoding and distance buckets
- Detects feature drift with a two-sample Kolmogorov–Smirnov test
- Logs every inference to SQLite (dev) or PostgreSQL (prod) for auditing
- Retrains automatically when 30 days of fresh data accumulate

---

## Setup

### Local

```bash
git clone https://github.com/atharvadevne123/Logistics-Flow
cd Logistics-Flow

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env

uvicorn app.main:app --reload
```

The API is then available at `http://localhost:8000`, with interactive
OpenAPI docs at `http://localhost:8000/docs`.

On first start the service trains a model on synthetic data and writes
`model.joblib`, `feature_pipeline.joblib`, and `metrics.json`.

### Docker

```bash
docker compose up --build
```

This starts the API on port 8000 alongside a PostgreSQL 15 instance, with the
model baked into the image at build time.

---

## API Reference

All endpoints are versioned under `/api/v1`.

### `POST /api/v1/predict`

Predict delivery time for a single shipment.

**Request**

```json
{
  "carrier": "DHL",
  "distance_km": 42.5,
  "weight_kg": 3.2,
  "route_type": "urban",
  "hour_of_day": 14,
  "day_of_week": 2
}
```

| Field | Type | Constraint |
|---|---|---|
| `carrier` | string | one of `DHL`, `FedEx`, `UPS`, `USPS`, `Amazon` |
| `distance_km` | float | `0 < x <= 5000` |
| `weight_kg` | float | `0 < x <= 100` |
| `route_type` | string | one of `urban`, `suburban`, `rural`, `highway` |
| `hour_of_day` | int | `0–23` |
| `day_of_week` | int | `0` (Mon) – `6` (Sun) |

**Response**

```json
{
  "predicted_minutes": 87.34,
  "predicted_hours": 1.456,
  "confidence": 0.9127,
  "model_version": "1.0.0",
  "request_id": "a3f9c1e2"
}
```

Invalid carriers, route types, or out-of-range numerics return `422`.

### `POST /api/v1/predict/batch`

Scores between 1 and 100 shipments in a single request:

```json
{ "shipments": [ { "carrier": "DHL", "distance_km": 42.5, "weight_kg": 3.2,
                   "route_type": "urban", "hour_of_day": 14, "day_of_week": 2 } ] }
```

Responds with `{ "predictions": [...], "count": n }`. One invalid member
rejects the whole batch with `422`.

### `GET /api/v1/health`

Liveness probe. Returns `healthy` when the model is loaded, `degraded` otherwise.

### `GET /api/v1/metrics`

Returns the most recent 5-fold cross-validation metrics: `rmse_mean`,
`r2_mean`, `n_features`, `n_samples`, `model_version`.

### `GET /api/v1/drift`

Runs a KS test comparing the last 100 predictions against the reference
window for `distance_km`, `weight_kg`, and `predicted_minutes`. Drift is
flagged when `p < 0.05`.

Every response carries `X-Request-ID` and `X-Response-Time-Ms` headers.

---

## Architecture

```
Client → FastAPI → Pydantic validation → Feature pipeline
                                              ↓
                        Ensemble (XGBoost + LightGBM + RandomForest)
                                              ↓
                        SQLAlchemy persistence → KS drift monitor
                                              ↓
                              Airflow weekly retrain DAG
```

### Feature engineering

Thirteen features are derived from six raw inputs:

- **Cyclical time** — `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` so that
  hour 23 sits adjacent to hour 0 rather than 23 units away
- **Flags** — `is_weekend`, `is_peak` (07–09 and 16–19)
- **Route** — `distance_bucket` (local/regional/long-haul/extreme),
  `weight_per_km`, `carrier_risk`, `route_code`
- **Encoding** — `carrier_enc` via label encoding

### Model

A `VotingRegressor` over XGBoost (200 trees, depth 5), LightGBM (200 trees),
and RandomForest (150 trees, depth 8), wrapped in a `StandardScaler` pipeline.
Confidence is derived from the standard deviation across sub-estimator
predictions — tight agreement yields high confidence.

### Monitoring

Every prediction is written to the `predictions` table. A rolling 500-sample
reference buffer feeds the KS test; detected drift is recorded in `drift_logs`
and surfaced through `/api/v1/drift`.

---

## Testing

```bash
pytest tests/ -v
```

The suite covers API contracts (including all carriers and route types via
parametrization), feature-pipeline invariants, model training and CV metrics,
and drift detection under known distribution shifts.

## Development

```bash
make install    # install dependencies
make test       # run pytest
make lint       # ruff check
make run        # start dev server
```

## License

MIT
