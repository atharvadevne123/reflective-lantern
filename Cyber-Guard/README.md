# Cyber-Guard

[![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Real-time network intrusion detection and cybersecurity threat classification API** using an XGBoost + RandomForest ensemble with KS-test drift monitoring, automated weekly retraining via Airflow, and full PostgreSQL persistence.

## Overview

Cyber-Guard classifies network connections into five threat categories:

| Class | Description |
|-------|-------------|
| `normal` | Legitimate traffic |
| `dos` | Denial-of-service attack |
| `probe` | Network reconnaissance / port scan |
| `r2l` | Remote-to-local unauthorized access |
| `u2r` | User-to-root privilege escalation |

## Architecture

![Architecture](screenshots/architecture.png)

## Tech Stack

- **API**: FastAPI with correlation-ID middleware, Pydantic v2 validation
- **ML**: XGBoost + RandomForest VotingClassifier (soft voting), 5-fold cross-validation
- **Features**: 15 engineered features — byte ratios, log transforms, rolling stats, protocol encodings
- **Monitoring**: SciPy KS-test drift detection on `src_bytes` distribution
- **Storage**: SQLAlchemy (SQLite dev / PostgreSQL prod) for prediction & drift logs
- **Retraining**: Airflow DAG scheduled weekly, accuracy gate ≥ 70%
- **Infra**: Docker + docker-compose, GitHub Actions CI (ruff + pytest)

## Quick Start

```bash
# 1. Clone & configure
git clone https://github.com/atharvadevne123/reflective-lantern.git
cd reflective-lantern/Cyber-Guard
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally (SQLite)
DATABASE_URL=sqlite:///./cyber_guard.db uvicorn app.main:app --reload

# 4. Or run with Docker (PostgreSQL)
docker-compose up --build
```

## API Reference

### POST /api/v1/predict

Classify a network connection.

**Request:**
```json
{
  "src_bytes": 491,
  "dst_bytes": 0,
  "duration": 0.0,
  "protocol_type": "tcp",
  "service": "http",
  "flag": "SF"
}
```

**Response:**
```json
{
  "prediction": "normal",
  "confidence": 0.9241,
  "class_probabilities": {
    "normal": 0.9241,
    "dos": 0.0312,
    "probe": 0.0218,
    "r2l": 0.0154,
    "u2r": 0.0075
  }
}
```

### GET /api/v1/health

Returns API and model status.

### GET /api/v1/metrics?hours=24&run_drift=false

Returns prediction statistics and optional drift check.

### GET /api/v1/drift

Runs KS-test drift check on `src_bytes` distribution.

## Environment Variables

See [.env.example](.env.example) for all variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./cyber_guard.db` | DB connection string |
| `MODEL_PATH` | `model.joblib` | Path to serialised model |
| `DRIFT_P_THRESHOLD` | `0.05` | KS-test p-value threshold |
| `REFERENCE_WINDOW_DAYS` | `7` | Days of history for reference distribution |

## Running Tests

```bash
cd Cyber-Guard
pytest tests/ -v --tb=short
```

## Feature Engineering

Fifteen features are derived from six raw packet fields:

- **Categorical encodings**: protocol, service, flag (label-encoded)
- **Ratio features**: `byte_ratio`, `total_bytes`, `src_dst_diff`
- **Log transforms**: `log_src_bytes`, `log_dst_bytes`, `log_duration`
- **Interaction features**: `bytes_per_second`
- **Rolling statistics**: 5-connection rolling mean and std of `src_bytes`

## Model Retraining

The Airflow DAG (`pipelines/retrain_dag.py`) runs weekly and:
1. Fetches the last 7 days of prediction logs
2. Retrains the ensemble model
3. Validates accuracy ≥ 70% (raises an error and prevents promotion otherwise)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
