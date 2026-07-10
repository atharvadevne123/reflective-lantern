# Watt-Guard ⚡

> Smart building and industrial energy consumption forecasting and anomaly detection API.

[![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/reflective-lantern/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)

## Overview

Watt-Guard provides production-grade energy consumption forecasting and anomaly detection for smart buildings and industrial facilities. It uses an XGBoost + LightGBM + RandomForest voting ensemble trained on occupancy-aware, weather-correlated features with automated weekly retraining and KS-test drift monitoring.

## Architecture

![Architecture](screenshots/architecture.png)

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ML Models | XGBoost, LightGBM, RandomForest (VotingRegressor) |
| Anomaly Detection | IsolationForest |
| Feature Pipeline | sklearn Pipeline (7 stages, 30+ features) |
| Drift Monitoring | KS-test (scipy) |
| Database | SQLite (dev) / PostgreSQL (prod) via SQLAlchemy |
| Retraining | Airflow weekly DAG |
| Containerisation | Docker + docker-compose |
| Testing | pytest (50+ tests) |
| CI | GitHub Actions (ruff + pytest) |

## Quickstart

```bash
# Clone
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern && git checkout innovation/watt-guard

# Install
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker-compose up --build
```

## API Reference

### POST `/api/v1/predict`
Forecast energy consumption for a building.

```json
{
  "building_id": "bldg-001",
  "timestamp": "2025-06-01T14:00:00",
  "hour": 14,
  "day_of_week": 1,
  "month": 6,
  "temperature_c": 28.5,
  "humidity_pct": 60.0,
  "occupancy": 50,
  "hvac_state": 1,
  "consumption_kwh": 12.0
}
```

### POST `/api/v1/anomaly`
Detect anomalous consumption readings.

### POST `/api/v1/drift`
KS-test between reference and current distributions.

### GET `/api/v1/metrics`
Prediction counts, anomaly counts, drift events, model R2/MAE.

### GET `/api/v1/health`
API liveness and model load status.

### POST `/api/v1/train`
Train both models on synthetic seed data (for demo/CI).

## Feature Engineering

The 7-stage sklearn Pipeline computes:
- **Temporal**: hour/day/month cyclic sin-cos, is_weekend, is_business_hour
- **Lag**: 1h, 2h, 3h, 6h, 12h, 24h, 168h consumption lags
- **Rolling**: mean/std/min/max over 3h, 6h, 24h windows
- **Weather**: heat index, cooling/heating degree-hours, temp-humidity ratio
- **Occupancy**: occ×HVAC load proxy, log-occupancy density
- **DropNonNumeric**: removes non-numeric columns before scaling
- **StandardScaler**: zero-mean unit-variance normalisation

## Model Monitoring

Every prediction is logged to `prediction_logs`. Drift is checked via KS-test between a reference window (last 500 readings after training) and the current window. Anomaly scores from IsolationForest are logged to `anomaly_logs`.

## Retraining

The Airflow DAG `watt_guard_weekly_retrain` runs every Sunday. It fetches recent data, validates row count ≥ 500, retrains the model, gates on R2 ≥ 0.70, and deploys the new artefact.

## Testing

```bash
pytest tests/ -v
```

## License

MIT
