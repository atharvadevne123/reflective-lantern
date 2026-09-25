# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-09-25

### Added

- mypy integration: `[tool.mypy]` config in `pyproject.toml`, `mypy>=1.10.0`
  in dev extras, and a `continue-on-error` type-check step in CI
- `pytest-cov>=5.0.0` in dev extras; CI now reports per-module coverage with
  `--cov=app --cov-report=term-missing`
- `functools.lru_cache` on `get_settings()` to avoid redundant env reads
- `__version__`, `__author__`, and `__description__` module-level metadata to
  `app/__init__.py`
- Responsible disclosure policy section to `SECURITY.md`
- Parametrized edge-case tests in `test_health_check.py`,
  `test_correlation_id.py`, and `test_middleware.py`
- `AlertRule.__post_init__` validation that rejects negative `cooldown_s`
- `CircuitBreaker.is_closed` and `is_half_open` properties for state queries
- `Page.item_count` property returning the number of items on a page
- `TTLCache.__bool__` and `TTLCache.get_or_set` convenience methods
- `TokenBucket.fill_ratio` property exposing fractional capacity remaining
- `EventBus.__len__` reporting total handler count (specific + wildcard)
- `TaskQueue.is_empty` property for empty-queue checks
- `add_file_handler()` helper in `logging_config` using `pathlib.Path.mkdir`
- `export_lineage_json()` helper in `data_versioning` serialising to JSON
- `buffer_stats()` in `monitoring` exposing reference-buffer fill levels
- Input-validation and circuit-breaker constant groups in `app/constants.py`
- `.python-version` file pinning the runtime to Python 3.11
- Max-length enforcement (`MAX_REGION_ID_LENGTH=64`, `MAX_EMAIL_LENGTH=254`)
  in `validate_region_id` and `validate_email`
- `__slots__` on `CircuitBreaker` and `slots=True` on `TokenBucket` dataclass
- Parametrized test suites for `retry`, `circuit_breaker`, `cache`,
  `pagination`, `token_bucket`, `event_bus`, `alerting`, and `validation`

### Changed

- Expanded module docstring in `app/constants.py` to describe each constant
  group
- Improved Google-style docstrings across `webhook_handler.py`,
  `notification_dispatcher.py`, and `shadow_mode.py`
- Expanded module-level docstrings in `retry`, `circuit_breaker`, `cache`,
  `token_bucket`, `event_bus`, and `task_queue`

### Fixed

- Resolved all ruff lint and format errors blocking CI

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
