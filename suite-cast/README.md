# Suite-Cast

![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Hotel booking demand forecasting and dynamic pricing API.** Suite-Cast predicts booking demand probability for a given room-night using an XGBoost + LightGBM ensemble, then converts that demand score into a dynamically priced room-rate suggestion. Every prediction is logged to a database, and a KS-test drift monitor compares live score distributions against the training reference so operators know when to retrain.

## Why

Hotels leave revenue on the table with static rate cards. Demand for a room-night varies with lead time, seasonality, day of week, occupancy trends, competitor pricing, and local events. Suite-Cast models these signals and returns a demand tier (low / medium / high) plus a suggested rate in a 0.7×–1.6× band around your base rate — the standard revenue-management envelope.

## Features

- **Ensemble model** — XGBoost + LightGBM averaged, each validated with 5-fold CV (AUC-ROC)
- **18-feature sklearn Pipeline** — lead-time buckets, seasonality index, competitor rate ratio, YoY occupancy delta, weekend×summer interaction, ordinal encodings
- **Dynamic pricing** — demand score drives a bounded price multiplier on the base rate
- **Model monitoring** — every prediction persisted via SQLAlchemy; `/metrics` runs a KS-test drift check against the training reference distribution
- **Automated retraining** — Airflow DAG with champion/challenger promotion gated on AUC improvement
- **Production API hygiene** — Pydantic validation, rate limiting, correlation-ID tracing, structured JSON logs, OpenAPI docs
- **Docker-first** — single `docker compose up` brings up the API and PostgreSQL

## Quick Start

```bash
# Clone and enter the project
cd suite-cast

# Install
make install          # or: pip install -r requirements.txt

# Run tests
make test

# Start the dev server (SQLite, auto-trains on first boot)
make run
# → http://localhost:8000/docs
```

### Docker (production-style)

```bash
docker compose up -d --build
# API on :8000, PostgreSQL on :5432
```

### Configuration

Copy `.env.example` to `.env` and adjust. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./suite_cast.db` | SQLAlchemy connection string |
| `BASE_ROOM_RATE` | `150.0` | Baseline rate for dynamic pricing |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP sliding-window limit |
| `MODEL_VERSION` | `1.0.0` | Version tag stamped on predictions |

## API Reference

### `POST /api/v1/predict`

Predict demand and get a suggested rate.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lead_time": 14,
    "length_of_stay": 3,
    "guests_count": 2,
    "checkin_month": 7,
    "checkin_dayofweek": 4,
    "is_weekend": 1,
    "current_occ_rate": 0.75,
    "prev_year_occ_rate": 0.70,
    "room_rate": 189.0,
    "competitor_avg_rate": 175.0,
    "special_event": 0,
    "room_type": "deluxe",
    "booking_channel": "online"
  }'
```

Response:

```json
{
  "request_id": "5f0c4d6e-...",
  "demand_score": 0.7412,
  "demand_tier": "high",
  "suggested_rate": 226.51,
  "model_version": "1.0.0"
}
```

### `GET /api/v1/health`

Liveness + model readiness + training AUC.

### `GET /api/v1/metrics`

Aggregated prediction stats and the current drift report:

```json
{
  "prediction_stats": {"count": 128, "avg_demand_score": 0.61, ...},
  "drift": {"ks_statistic": 0.08, "p_value": 0.42, "drift_detected": false},
  "model_version": "1.0.0"
}
```

