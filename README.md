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
