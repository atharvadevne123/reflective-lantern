# Changelog

All notable changes to Temporal-Pulse are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-08

### Added

- FastAPI application with 8 endpoints: `/detect`, `/forecast`, `/train`,
  `/health`, `/metrics`, `/version`, `/drift`, `/feature-importance`
- Isolation Forest anomaly detector with RobustScaler preprocessing
- Random Forest multi-step forecaster with tree-level confidence intervals
- FAISS nearest-neighbour anomaly root cause analysis (sklearn fallback)
- Feature engineering pipeline: rolling statistics (3 windows), lag features
  (4 steps), rate of change, cyclical time encoding, cross-sensor correlations
- KS-test drift detection with per-feature reference/current distributions
- Prediction logging with latency percentiles (p95/p99)
- SQLAlchemy models: sensor_readings, anomaly_events, predictions, drift_logs
- Airflow-compatible daily retraining DAG
- Docker + docker-compose with PostgreSQL 16
- GitHub Actions CI: ruff lint, pytest with coverage, mypy type check
- Correlation ID middleware with request latency headers
- Test suite with in-memory SQLite fixtures (50+ tests)

### Security

- Pydantic v2 input validation on all endpoints (NaN/Inf rejection)
- Environment-based configuration; no hardcoded credentials
