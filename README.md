# Traffic-Pulse

[![CI](https://github.com/atharvadevne123/Traffic-Pulse/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/Traffic-Pulse/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688)
![License](https://img.shields.io/badge/license-MIT-green)

Real-time urban traffic congestion prediction and incident detection API using an
**XGBoost + LightGBM ensemble** with temporal feature engineering, route scoring,
and automated KS-test drift detection.

Traffic-Pulse classifies road segments into four congestion levels — `free`,
`moderate`, `congested`, `severe` — from live traffic telemetry (vehicle counts,
average speeds, incidents, weather), logs every prediction for monitoring, and
retrains automatically when feature drift is detected.

## Features

- **Ensemble ML** — XGBoost + LightGBM pipelines averaged at the probability level,
  trained with 5-fold stratified cross-validation (weighted one-vs-rest AUC-ROC).
- **26 engineered features** — cyclical hour/day-of-week encodings, peak-hour flags,
  lag features (1h/2h/4h), rolling means/stds (6h/24h), speed-volume ratios,
  incident density, and road-type encoding.
- **Drift detection** — two-sample Kolmogorov-Smirnov test per feature with
  drift events persisted to the database and surfaced in `/api/v1/metrics`.
- **Prediction logging** — every request/result stored via SQLAlchemy
  (SQLite in dev, PostgreSQL in prod).
- **Automated retraining** — drift-gated pipeline with AUC validation and
  model promotion (`pipelines/retrain_dag.py`, Airflow-compatible).
- **Production middleware** — correlation-ID propagation and response-time headers.

## Quickstart

### Local

```bash
git clone https://github.com/atharvadevne123/Traffic-Pulse
cd Traffic-Pulse
make install          # pip install -r requirements.txt + dev tools
make train            # train the ensemble (writes model.joblib + metrics.json)
make run              # uvicorn app.main:app --reload
```

### Docker

```bash
cp .env.example .env
docker compose up --build -d
curl http://localhost:8000/health
```

## API Reference

### `POST /api/v1/predict`

Predict the congestion level for a route segment.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "route_id": "R42",
    "hour": 8,
    "day_of_week": 1,
    "vehicle_count": 1800,
    "avg_speed_kmh": 32.5,
    "road_type": "arterial",
    "incident_count": 1,
    "is_raining": 1
  }'
```

```json
{
  "route_id": "R42",
  "congestion_level": 2,
  "congestion_label": "congested",
  "congestion_probability": 0.71,
  "class_probabilities": {"free": 0.02, "moderate": 0.21, "congested": 0.71, "severe": 0.06},
  "incident_score": 0.0556,
  "model_version": "1.0.0"
}
```

### `POST /api/v1/drift`

Run a KS test between a reference window and the current window of a feature.

```bash
curl -X POST http://localhost:8000/api/v1/drift \
  -H "Content-Type: application/json" \
  -d '{"feature_name": "vehicle_count", "reference": [/* >=10 floats */], "current": [/* >=10 floats */]}'
```

### `GET /api/v1/metrics`

Prediction volume, congestion distribution, active drift alerts, and training metrics.

### `GET /health`

Liveness probe with model version and loaded ensemble members.

