# Alpha-Scout

[![CI](https://github.com/atharvadevne123/Alpha-Scout/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/Alpha-Scout/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io)

**ML-powered portfolio alpha generation system.** Alpha-Scout scores securities on 20+ technical and fundamental signals, predicts stock return potential, ranks portfolios, and serves real-time allocation recommendations via a FastAPI REST API.

## Overview

Alpha-Scout ingests OHLCV (open/high/low/close/volume) data for individual securities, engineers a rich feature set covering momentum, volatility, mean-reversion and fundamental factors, then scores each security with an XGBoost ensemble trained on 5-fold cross-validated historical patterns. Each prediction is logged to a database, and a continuous KS-test drift monitor alerts when input distributions shift.

## Architecture

![Architecture](screenshots/architecture.svg)

| Layer | Technology |
|-------|------------|
| API | FastAPI + Uvicorn (8+ endpoints, OpenAPI docs) |
| ML Model | XGBoost + sklearn Pipeline + 5-fold CV |
| Features | 29 signals: RSI, MACD, rolling stats, momentum lags, ratios |
| Monitoring | KS-test drift detection, prediction logging |
| Persistence | SQLAlchemy ORM (SQLite dev / PostgreSQL prod) |
| Infrastructure | Docker + docker-compose + PostgreSQL |
| CI/CD | GitHub Actions (ruff lint + pytest) |
| Retraining | Airflow-compatible DAG (weekly schedule) |

## Quick Start

```bash
git clone https://github.com/atharvadevne123/Alpha-Scout
cd Alpha-Scout
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`

### Docker

```bash
docker-compose up --build
```

## API Reference

### POST `/api/v1/predict` — Score a security
### GET `/api/v1/health` — Service health
### GET `/api/v1/metrics` — Aggregate prediction stats
### POST `/api/v1/drift` — KS-test drift check
### POST `/api/v1/rank` — Rank a portfolio of securities
### GET `/api/v1/feature-importance` — Top 10 XGBoost feature importances
### POST `/api/v1/anomaly` — Isolation Forest anomaly detection
### GET `/api/v1/pipeline-health` — ML pipeline diagnostics
### GET `/api/v1/predictions/export` — Export recent predictions as JSON

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

63 tests, all passing. Ruff: clean (0 errors).

## Retraining

```bash
python -c "from pipelines.retrain_dag import run_pipeline; run_pipeline()"
```

## Environment Variables

See `.env.example` for all configuration options.

## Model Quality

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Classifier |
| Cross-validation | 5-fold stratified |
| Scoring | ROC-AUC |
| Features | 29 engineered signals |
| Training tickers | 5 synthetic OHLCV series |
