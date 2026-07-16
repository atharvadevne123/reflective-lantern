# Changelog

All notable changes to Suite-Cast are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/).

## [1.0.0] — 2026-07-15

### Added

- XGBoost + LightGBM ensemble demand model with 5-fold cross-validated AUC-ROC.
- 18-feature sklearn Pipeline: lead-time buckets, seasonality index, competitor rate ratio, YoY occupancy delta, weekend×summer interaction, ordinal encodings.
- FastAPI service with `/api/v1/predict`, `/api/v1/health`, `/api/v1/metrics`.
- Dynamic pricing engine mapping demand score to a 0.7×–1.6× rate multiplier.
- SQLAlchemy prediction logging (SQLite dev / PostgreSQL prod) with request UUIDs.
- KS-test drift detection comparing production scores to training reference.
- Airflow retraining DAG with champion/challenger promotion gate.
- Sliding-window rate limiting and correlation-ID middleware.
- Structured JSON logging.
- Alembic migration environment.
- Docker + docker-compose (API + PostgreSQL 15) with healthchecks.
- pytest suite covering API, model, features, and monitoring.
- GitHub Actions CI: ruff lint, format check, pytest.
- Architecture diagram generator (matplotlib).
