# Changelog

All notable changes to Forge-Guard are documented here.

## [1.1.0] — 2026-08-04

### Added
- `GET /api/v1/version` endpoint — returns api_version, model_version, python_version, service name
- `POST /api/v1/anomaly` endpoint — FAISS nearest-neighbour anomaly check
- `GET /api/v1/export/predictions` and `GET /api/v1/export/drift` export endpoints
- `app/config.py` — centralised Settings with `lru_cache` singleton
- `app/cache.py` — thread-safe TTL prediction cache (30 s default, 2048 max entries)
- `app/validators.py` — domain-range validation and `sanitize_sensor_reading` helper
- `app/middleware.py` — `RequestTimingMiddleware` (X-Process-Time) and `SecurityHeadersMiddleware`
- `app/reporting.py` — CSV/JSON export utilities for predictions and drift reports
- `compute_zscore_outliers` and `model_prediction_summary` in `monitoring.py`
- `DriftCheckResult` and `ModelSummaryResponse` Pydantic schemas
- Database indexes on `timestamp`, `prediction`, `model_version`, and `correlation_id` columns
- `lru_cache` on model metrics disk read (`_read_metrics_cached`)
- Google-style docstrings across all public API functions
- New Makefile targets: `install-dev`, `test-cov`, `typecheck`, `healthcheck`, `seed`
- `scripts/healthcheck.py` — probes `/health`, exits 0/1
- `SECURITY.md` and `CODE_OF_CONDUCT.md` community health files
- 15 new test modules covering cache, config, validators, middleware, reporting, and pipeline

### Changed
- `load_model` raises `RuntimeError` (instead of silent fallback) on corrupt model file
- `conftest.py` extended with `low_risk_payload`, `boundary_payload`, `large_synthetic_df` fixtures

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
