# Temporal-Pulse

[![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/reflective-lantern/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Multivariate Time-Series Anomaly Detection and Forecasting API** — built with FastAPI, Isolation Forest + Random Forest ensemble, FAISS nearest-neighbour root cause analysis, KS-test drift detection, and a PostgreSQL backend.

## Overview

Temporal-Pulse ingests streaming sensor readings, extracts a rich feature set (rolling statistics, lag features, rate-of-change, cyclical time encoding, and cross-sensor correlations), scores each observation for anomalies, and returns multi-step forecasts with confidence intervals. All anomaly events are persisted and indexed for fast similarity search.

**Key capabilities:**
- Real-time anomaly scoring with configurable threshold
- Multi-step forecasting with per-tree confidence intervals
- FAISS-based root cause explanation (nearest historical anomalies)
- KS-test data drift monitoring across all features
- Automated daily retraining pipeline (Airflow-compatible)
- Full OpenAPI documentation at `/docs`

## Quick Start

### Local (SQLite)

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern/temporal-pulse
pip install -r requirements.txt
cp .env.example .env
# Edit .env to use SQLite:
# DATABASE_URL=sqlite:///./temporal_pulse.db
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Docker Compose (PostgreSQL)

```bash
cd temporal-pulse
cp .env.example .env   # adjust DATABASE_URL if needed
docker compose up -d
```

### Run Tests

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v --cov=app
```

### Detect Anomalies (cURL)

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [{
      "sensor_id": "turbine-01",
      "timestamp": "2026-07-08T12:00:00",
      "values": {"vibration": 9.8, "temperature": 95.1, "rpm": 3200}
    }],
    "horizon": 5
  }'
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/v1/health` | Liveness + readiness check |
| `GET`  | `/api/v1/version` | API and model version |
| `GET`  | `/api/v1/metrics` | Prediction and latency metrics |
| `POST` | `/api/v1/train` | Train anomaly detector and forecaster |
| `POST` | `/api/v1/detect` | Batch anomaly detection |
| `POST` | `/api/v1/forecast` | Multi-step forecasting |
| `GET`  | `/api/v1/drift` | KS-test drift report |
| `GET`  | `/api/v1/feature-importance` | Top-20 RF feature importances |

### POST /api/v1/detect — Request Body

```json
{
  "readings": [
    {
      "sensor_id": "string",
      "timestamp": "ISO-8601",
      "values": { "channel_name": 0.0 }
    }
  ],
  "horizon": 5
}
```

### POST /api/v1/detect — Response

```json
{
  "results": [
    {
      "sensor_id": "turbine-01",
      "timestamp": "2026-07-08T12:00:00",
      "anomaly_score": 0.87,
      "is_anomaly": true,
      "threshold": 0.7
    }
  ],
  "total_readings": 1,
  "anomaly_count": 1,
  "processing_time_ms": 12.4
}
```
