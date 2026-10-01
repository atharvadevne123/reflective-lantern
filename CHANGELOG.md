# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-10-01 Improvement Run

### Added

- `GET /api/v1/version` endpoint returning build metadata and Python version
- `app/version.py` module with structured version metadata
- Return type annotations on all async endpoints in `app/main.py`
- Return type annotation on `RateLimitMiddleware.dispatch`
- Google-style docstrings to `app/benchmarks.py`, `app/circuit_breaker.py`,
  `app/compression.py`, `app/cost_estimator.py`, `app/cache.py`,
  `app/features.py`, `app/database.py`, `app/exceptions.py`, `app/main.py`,
  `app/alerting.py`, `app/anomaly.py`, `app/battery.py`, `app/forecasting.py`,
  and `app/carbon.py`
- Enum-member docstrings on `CircuitState` values (CLOSED, OPEN, HALF_OPEN)
- 100+ new test functions across 20 test files covering edge cases for
  webhook handler, summarize_history, experiment tracker, FAISS index,
  feature store, model registry, shadow mode, and many other modules
- `README.md` Features section listing all major capabilities
- `README.md` API Reference entry for `GET /api/v1/version`
- `README.md` Testing section with test count (92 modules, 6 000+ tests)
  and coverage command
- `pyproject.toml` mypy configuration block
- `.github/CODEOWNERS` file
- `app/__init__.py` explicit `__all__` list exporting main public symbols
- `DispatchResult` dataclass field-level docstrings in `app/battery.py`
- Expanded Args/Returns/Note sections in `app/forecasting.py`, `app/carbon.py`,
  `app/anomaly.py`, and `app/alerting.py` for previously one-liner docstrings
- Edge-case tests to `test_foundry_sync.py`, `test_pipeline.py`,
  `test_email_report.py`, `test_middleware.py`, `test_notification_dispatcher.py`,
  `test_exceptions.py`, `test_profiler.py`, `test_mode.py`,
  `test_monitoring_extended.py`, `test_compression.py`, and `test_circuit_breaker.py`

### Changed

- All `print()` calls in `scripts/` replaced with `logging` calls for
  consistent structured log output (9 scripts updated)

### Fixed

- Resolved 59 ruff lint and format errors across 125 files (E, F, I, UP, B,
  SIM rule sets)

## [1.0.0] - 2026-08-20

### Added

- FastAPI service with versioned `/api/v1` endpoints: `predict`, `health`,
  `metrics`, and `drift`
- Ensemble regression model combining XGBoost, LightGBM, and RandomForest
  under a `StandardScaler` pipeline
- Feature engineering pipeline producing 13 features from 6 raw inputs,
  including cyclical time encoding, distance bucketing, weight-per-km ratio,
  and carrier risk scoring
- Ensemble-spread confidence scoring on every prediction
- KS-test drift detection with a rolling 500-sample reference buffer
- SQLAlchemy persistence for predictions and drift logs (SQLite dev,
  PostgreSQL prod)
- Airflow DAG for weekly automated retraining with a 200-row minimum guard
- Correlation-ID middleware emitting `X-Request-ID` and `X-Response-Time-Ms`
- Pydantic validation rejecting unknown carriers, route types, and
  out-of-range numerics
- Docker and docker-compose setup with PostgreSQL 15 and healthchecks
- pytest suite covering API, features, model, and monitoring
- GitHub Actions CI running ruff lint, format check, and tests
- Architecture diagram generator
- Bulk scoring endpoint `POST /api/v1/predict/batch` (1-100 shipments)
- Sliding-window rate limiting with `X-RateLimit-*` and `Retry-After` headers
- Environment-driven settings module and typed domain exceptions
- Alembic migration environment with the initial schema revision
- Deployment guide, API reference, smoke test, and latency benchmark

### Fixed

- Feature pipeline reported itself unfitted because the encoder stored its
  `LabelEncoder` as `_le`; sklearn's `check_is_fitted` only recognises
  attributes *ending* in an underscore. Renamed to `le_`.
- Prediction confidence was hardcoded to 1.0 by an expression that always
  evaluated to zero. It is now derived from ensemble sub-estimator spread.
- Unseen carriers raised at transform time instead of falling back gracefully.
