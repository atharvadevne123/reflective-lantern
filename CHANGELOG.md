# Changelog

All notable changes to Traffic-Pulse are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Planned

- Redis-backed rate limiting for multi-replica deployments.
- MLflow experiment tracking for retraining runs.
- Real-world data connectors (city open-data traffic feeds).

## [1.0.0] - 2026-07-08

### Added

- XGBoost + LightGBM ensemble classifier for 4-level congestion prediction
  (free / moderate / congested / severe) with 5-fold stratified CV and
  weighted one-vs-rest AUC-ROC reporting.
- Feature engineering pipeline with 26 features: cyclical hour/day encodings,
  peak-hour flags, lag features (1h/2h/4h), rolling statistics (6h/24h),
  speed-volume ratios, incident density, and road-type encoding.
- FastAPI service with versioned endpoints: `POST /api/v1/predict`,
  `POST /api/v1/drift`, `GET /api/v1/metrics`, `GET /health`.
- KS-test drift detection with per-feature drift logging to the database.
- Prediction logging via SQLAlchemy (SQLite dev, PostgreSQL prod).
- Automated retraining pipeline (`pipelines/retrain_dag.py`) with
  drift-gated retraining, AUC validation, and model promotion.
- Correlation-ID and response-time middleware.
- Docker + docker-compose with PostgreSQL 16 and healthchecks.
- GitHub Actions CI: ruff lint + format check + pytest.
- Batch prediction endpoint (`POST /api/v1/predict/batch`, max 100 items).
- Per-route history endpoint (`GET /api/v1/routes/{route_id}/history`).
- Model metadata endpoint (`GET /api/v1/model-info`).
- Rate limiting (120 req/min per client) and gzip compression.
- Structured JSON logging with correlation IDs.
- Alembic migrations for prediction and drift tables.
- Test suite: 85+ tests across API, model, features, monitoring, middleware, and pipeline.
