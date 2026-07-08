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
