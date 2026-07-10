# Changelog

All notable changes to Forge-Guard are documented here.

## [1.0.0] — 2026-06-24

### Added
- FastAPI service with `/api/v1/predict`, `/health`, `/api/v1/metrics` endpoints
- XGBoost + RandomForest soft-voting ensemble with 5-fold stratified CV
- sklearn feature pipeline: lag, rolling stats, ratio, polynomial features
- KS-test drift detection across all 7 sensor features
- Prediction logging to SQLite (dev) / PostgreSQL (prod) via SQLAlchemy
- Automated retraining pipeline (`pipelines/retrain_dag.py`)
- Docker + docker-compose with API service and PostgreSQL
- GitHub Actions CI (ruff lint + pytest on Python 3.10/3.11/3.12)
- Rate limiting middleware (configurable RPM)
- Correlation ID middleware and header echo
- Pydantic v2 input validation with domain-specific field bounds
- API versioning (`/api/v1/...`) with legacy alias support
- Architecture diagram (`screenshots/architecture.png`)
- Comprehensive pytest suite (conftest, 5 test modules, parametrized cases)
- Alembic migration scaffolding
