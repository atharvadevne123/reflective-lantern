# Changelog

All notable changes to Fleet-Mind are documented here.

## [1.0.0] — 2026-10-28

### Added
- FastAPI application with `/api/v1/predict`, `/api/v1/predict/batch`, `/health`,
  `/api/v1/metrics`, `/api/v1/drift`, `/api/v1/retrain`, and `/api/v1/version` endpoints
- XGBoost + LightGBM voting ensemble for maintenance prediction
- 8-feature engineering pipeline (ratios, brake health, tyre deviation, battery delta)
- 5-fold stratified cross-validation with AUC-ROC reporting
- Remaining Useful Life (RUL) estimate based on brake pad thickness
- KS-test drift monitoring with configurable p-value threshold
- SQLAlchemy ORM with `VehiclePrediction` and `DriftReport` tables
- Correlation-ID and in-process rate-limiting ASGI middleware
- Airflow-compatible daily retraining DAG
- Structured logging with JSON output option
- Docker + docker-compose with PostgreSQL
- GitHub Actions CI workflow
- Comprehensive pytest suite (API, model, features, monitoring)
- `.env.example`, `pyproject.toml`, `.pre-commit-config.yaml`, `Makefile`
