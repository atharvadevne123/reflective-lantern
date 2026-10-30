# Fleet-Mind

[![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/fleet-mind-ci.yml/badge.svg)](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/fleet-mind-ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time fleet vehicle telemetry anomaly detection and predictive maintenance.  
Send 10 sensor readings per cycle — receive maintenance flags, anomaly type classification, and remaining-useful-life estimates.

---

## Overview

Fleet-Mind ingests raw vehicle telemetry (speed, engine temperature, oil pressure, brake wear, etc.), engineers 8 derived features, and runs them through an **XGBoost + LightGBM soft-voting ensemble** trained with **5-fold stratified cross-validation**.

A continuous **KS-test drift monitor** compares live feature distributions against training baselines and flags shifts before they degrade model accuracy.

---

## Quick Start

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd fleet-mind
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive API.

### Docker

```bash
docker-compose up --build -d
```

---

## API Reference

All prediction endpoints are under `/api/v1/`.

### POST `/api/v1/predict`

Predict maintenance need for a single vehicle.

**Request body**

```json
{
  "vehicle_id": "VH-001",
  "speed_kmh": 85.0,
  "engine_temp_c": 92.0,
  "oil_pressure_bar": 3.5,
  "fuel_consumption_lph": 7.2,
  "battery_voltage_v": 12.8,
  "brake_pad_mm": 6.5,
  "tire_pressure_bar": 2.3,
  "rpm": 2800.0,
  "coolant_temp_c": 88.0,
  "transmission_temp_c": 75.0
}
```

**Response**

```json
{
  "vehicle_id": "VH-001",
  "maintenance_needed": false,
  "anomaly_score": 0.12,
  "anomaly_type": null,
  "remaining_useful_life_km": 24000.0,
  "confidence": 0.76,
  "model_version": "1.0.0"
}
```

### POST `/api/v1/predict/batch`

Batch prediction for up to 100 vehicles.

```json
{ "readings": [ /* array of TelemetryInput objects */ ] }
```

### GET `/health`

Service liveness and DB connectivity check.

### GET `/api/v1/metrics`

Model performance metrics (accuracy, precision, recall, F1, AUC-ROC, CV-AUC, maintenance rate).

### GET `/api/v1/drift`

Run KS-test drift analysis on recent telemetry vs training baseline.

### POST `/api/v1/retrain`

Trigger model retraining on fresh data.

### GET `/api/v1/version`

Return API and model version strings.

---

## Architecture

```
Client
  │
  ▼
FastAPI (Correlation-ID + Rate-Limiter middleware)
  │
  ├── /api/v1/predict  ──▶  FeaturePipeline  ──▶  XGB+LGB Ensemble  ──▶  DB log
  ├── /api/v1/drift    ──▶  KS-test vs baseline  ──▶  DriftReport DB
  └── /api/v1/retrain  ──▶  train_model()  ──▶  overwrite model on disk
  │
  ├── PostgreSQL (VehiclePrediction, DriftReport)
  └── Joblib model file
```

**Feature Engineering** (18 total = 10 raw + 8 derived):

| Derived Feature | Formula |
|---|---|
| engine_heat_ratio | engine_temp / coolant_temp |
| oil_efficiency | oil_pressure / (rpm/1000) |
| fuel_speed_ratio | fuel_consumption / (speed + 1) |
| thermal_delta | transmission_temp − coolant_temp |
| power_index | rpm × speed / 1e6 |
| brake_health_score | brake_pad_mm / 12.0 |
| tire_pressure_deviation | abs(tire_pressure − 2.4) |
| battery_delta | abs(battery_voltage − 12.6) |

---

## Testing

```bash
make test
# or
pytest tests/ -v --cov=app
```

Test coverage includes API endpoints, model training/inference, feature engineering, and drift monitoring.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
