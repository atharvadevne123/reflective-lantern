# Changelog

## [2026-08-19] — Improvement Run

### Fixed
- CI: Resolved 7 ruff lint errors across `app/similarity.py`, `app/time_series.py`, `app/validation.py`, `app/date_utils.py`, `app/reporting.py`, `price-prophet/app/pricing/elasticity.py`, and `veritas-rag/app/scoring/trust.py` (B905, RUF002, RUF022)
- CI: Applied `ruff format` to 33 files to eliminate formatting drift

### Changed
- Added `-> None` return type annotations to test functions in 13 files across `tests/` and `cart-mind/tests/`, improving mypy compliance and code clarity


## [2026-07-20] — Improvement Run

### Changed
- Added `-> None` return type annotations to all test functions across 20 test files
- Added `__all__` export lists to 20+ app modules (carbon, date_utils, data_quality, forecasting, investment, reporting, stats_utils, trend_analysis, pipeline_utils, time_series, regions, benchmarks, cache, database, features, monitoring, validation, market_context, energy_export, notifications, middleware, mlflow_stub, logging_config, rate_limiter, config)
- Added `functools.lru_cache` to grid carbon intensity lookup in `app/carbon.py` for repeated region queries
- Improved docstring for `exponential_smoothing_forecast` with full Returns and Raises clauses
- Expanded parametrized test coverage in `test_forecasting.py`, `test_stats_utils.py`, `test_carbon.py`, `test_trend_analysis.py`, `test_data_quality.py`, `test_investment.py`, `test_date_utils.py`, `test_anomaly.py`, `test_reporting.py`, `test_pipeline_utils.py`, `test_regions.py`, `test_time_series.py`


## [1.2.0] - 2026-07-15

### Added
- `app/anomaly.py`: `batch_compute_severity()` and `anomaly_rate()` for bulk severity analysis
- `app/cache.py`: Thread-safe TTLCache with `RLock`, `__contains__`, and `stats()` method
- `app/validation.py`: `validate_building_id`, `validate_batch_size`, `validate_consumption_kwh`, `validate_feature_vector` input validators
- `app/regions.py`: Five new grid regions (pacific_nw, new_england, mountain, southeast, florida) plus `get_all_region_ids`, `get_region_timezone`, `get_peak_load` helpers
- `app/investment.py`: `mortgage_payment`, `roi_percentage`, `price_to_income_ratio` financial utilities
- `app/reporting.py`: `energy_efficiency_grade`, `monthly_consumption_summary` reporting helpers
- `app/market_context.py`: `rent_vs_buy_comparison`, `price_trend_indicator` market analysis utilities
- `app/similarity.py`: `euclidean_distance`, `batch_add` utility functions
- `app/mlflow_stub.py`: `set_tracking_uri`, `log_params`, `log_artifact`, `list_runs` stub helpers
- `app/model.py`: `get_feature_importance` to surface VotingRegressor feature importances
- `app/monitoring.py`: `get_anomaly_stats`, `compute_feature_drift_summary` monitoring aggregates
- `app/schemas.py`: `FeatureImportanceItem/Response`, `AnomalyStatsResponse`, `SavingsRequest/Response` schemas
- `app/main.py`: `/api/v1/feature-importance`, `/api/v1/anomaly/stats`, `/api/v1/savings`, `/api/v1/efficiency-grade` endpoints
- `app/database.py`: `ModelMetrics` ORM table; `get_predictions_by_building` query helper
- `app/features.py`: `InteractionFeatureExtractor` for pairwise feature products
- `app/faiss_index.py`: `reset()`, `__len__`, `save_vectors()` on `LoadPatternIndex`
- `app/aws_stub.py`: `delete_artefact` helper; corrected `__all__` exports
- `app/logging_config.py`: `TraceIdFilter`, `add_trace_id_filter` for per-request trace correlation
- `pipelines/retrain_dag.py`: `check_drift_before_retrain` KS-test gating step
- `scripts/summarize_history.py`: `export_to_csv` utility function
- `config/settings.py`: `to_dict`, `missing_optional` methods on `Settings`
- `config/constants.py`: ML/API/batch limit constants
- `config/mode.py`: `upcoming_innovation_days`, `mode_schedule` schedule helpers
- `.github/workflows/ci.yml`: Python 3.12 matrix; coverage XML artifact upload
- `.github/ISSUE_TEMPLATE/bug_report.md`: Bug report issue template
- `.github/ISSUE_TEMPLATE/feature_request.md`: Feature request issue template
- `Makefile`: `coverage-xml`, `profile` targets; `coverage.xml` in clean rule
- `pyproject.toml`: `[tool.coverage.run]` and `[tool.coverage.report]` configuration

## [1.1.0] - 2026-07-11

### Fixed
- `app/cache.py`: Removed undefined `default_ttl` parameter reference in `TTLCache.__init__`; cache misses on expired keys now correctly increment the `misses` counter
- `app/middleware.py`: Added missing `from collections import deque` import; moved `from typing import Any` to the correct top-of-file position; removed duplicate `_rate_buckets` global declaration
- `app/model.py`: Removed duplicate `XGBRegressor` import block; added missing `from typing import Any`; wrapped `joblib.load` calls in try/except with `logger.exception`
- `app/main.py`: Added missing `from typing import Any` (required by module-level `dict[str, Any]` annotations)
- `app/monitoring.py`: Added missing `from sqlalchemy.orm import Session` and `import numpy as np`
- `app/time_series.py`: Removed five lines of dead code after `return result.tolist()` in `simple_moving_average`
- `app/config.py`: Corrected default `DATABASE_URL` from `traffic_pulse.db` to `watt_guard.db`
- `app/logging_config.py`: Fixed module docstring (was "Volt-Cast", now "Watt-Guard")
- `tests/test_model.py`: Removed stray nested method `test_synthetic_loads_in_range` with undefined fixture
- `tests/test_anomaly.py`: Removed stray nested method `test_insufficient_reference` referencing undefined `quick_anomaly_check`
- `tests/test_time_series.py`: Removed stray nested method `test_peak_trough_indices` referencing undefined `seasonal_summary`
- `tests/test_cache.py`: Rewrote broken tests that used removed `default_ttl` param and non-existent `set(ttl=…)` overload
- `tests/test_features.py`: Added missing `import numpy as np` and `import pandas as pd`

### Added
- `app/main.py`: New `GET /api/v1/version` endpoint returning semantic version and build info
- `app/main.py`: New `GET /api/v1/regions` endpoint listing all supported grid regions
- `app/regions.py`: `@lru_cache(maxsize=32)` on `get_region()` to cache repeated region lookups
- `app/database.py`: PostgreSQL connection pooling via `DB_POOL_SIZE` and `DB_MAX_OVERFLOW` env vars; `index=True` on `PredictionLog.timestamp` and `PredictionLog.created_at`
- `app/reporting.py`: Length mismatch guard in `estimate_savings` raising `ValueError`
- `app/similarity.py`: `clear()` method on `BuildingSimilarityIndex`
- `Makefile`: `coverage`, `typecheck`, `check`, `migrate`, `clean`, `help` targets
- `.env.example`: PostgreSQL pool config, `LOG_JSON`, `AIRFLOW_HOME`, `R2_GATE`, `MIN_TRAINING_ROWS`
- `tests/test_reporting.py`: Parametrized tariff/cost tests and additional assertion coverage
- `tests/test_monitoring.py`: `test_latency_timer_ms_positive`, `test_compute_drift_exact_same_distribution`, `test_set_reference_window_truncates_to_500`
- `tests/test_api.py`: `test_batch_predict_after_train`, `test_batch_predict_exceeds_limit`, `test_drift_strong_shift_detected`

### Changed
- `app/schemas.py`: Added class-level docstrings to `PredictResponse`, `AnomalyResponse`, `DriftRequest`, `DriftResponse`
- `app/validation.py`: Narrowed return type of `extract_temporal_from_datetime` from `dict` to `dict[str, object]`
- `app/logging_config.py`: Added `correlation_id` propagation in `JsonFormatter.format`
- `pipelines/retrain_dag.py`: Added Google-style docstrings with Args/Returns/Raises to all DAG task functions
- `scripts/example_client.py`: Rewritten for Watt-Guard API with logging, type hints, and full docstrings
- `scripts/generate_diagram.py`: Type-annotated helper functions; replaced `print()` with `logger.info()`
- `tests/conftest.py`: Added docstrings to all fixtures; type-annotated `create_test_db`

## [1.0.0] - 2026-07-10

### Added
- FastAPI REST API with `/api/v1/predict`, `/api/v1/anomaly`, `/api/v1/drift`, `/api/v1/metrics`, `/api/v1/health`, `/api/v1/train`
- XGBoost + LightGBM + RandomForest `VotingRegressor` ensemble for energy consumption forecasting
- 7-stage sklearn feature engineering pipeline: temporal cyclic encoding, lag features (1h–168h), rolling stats (3h/6h/24h), weather features, occupancy features, drop-non-numeric, StandardScaler
- IsolationForest anomaly detector with severity classification
- KS-test drift detection with configurable reference window
- SQLAlchemy ORM: `EnergyReading`, `PredictionLog`, `AnomalyLog`, `DriftLog`
- PostgreSQL + Docker Compose production setup
- Airflow weekly retraining DAG with R2≥0.70 and row-count≥500 gates
- Rate limiting middleware (200 req/min per IP)
- Correlation ID middleware for distributed tracing
- pytest suite with 50+ tests and parametrized cases
- GitHub Actions CI (ruff lint + pytest)
- Architecture diagram

## [2026-08-03] — Improvement Run

### Added
- `app/trend_analysis.py`: `momentum_score()` (short vs long MA signal) and `cumulative_sum()` (CUSUM drift detection)
- `app/date_utils.py`: `datetime_to_iso()`, `start_of_day()`, `days_between()` utilities
- `app/carbon.py`: `carbon_budget_remaining()` for remaining CO2 allowance tracking
- `app/notifications.py`: `alert_summary()` and `highest_severity()` for alert aggregation
- `app/config.py`: `get_settings()`, `is_production()`, `effective_log_level()` config helpers
- `app/data_quality.py`: `field_value_counts()` and `null_rate()` for data profiling
- `app/energy_export.py`: `sort_records()` and `partition_records()` for record manipulation
- `app/pipeline_utils.py`: `pipeline_step_types()` and `first_step()` introspection helpers
- `energy_seer/app/validators.py`: `validate_forecast_length()`, `validate_tariff()`, `validate_readings_list()`
- `energy_seer/app/telemetry.py`: `increment_by()`, `all_above()`, `top_n()` counter utilities
- `energy_seer/app/grid_report.py`: `summarise_alerts()`, `report_status_code()`, `merge_reports()`
- `energy_seer/app/features.py`: `feature_column_names()`, `validate_dataframe_columns()`, `__all__`

### Fixed
- `app/anomaly.py`: Added missing `compute_z_score`, `flag_z_score_outliers`, `flag_anomaly_rate`, `consecutive_anomaly_runs`, `ewma_smooth` to `__all__`
- `app/stats_utils.py`: Removed duplicate `coefficient_of_variation` definition

### Tests
- Added 35+ new test cases across 15+ test files covering new and previously untested functions
- New test files: `energy_seer/tests/test_validators.py`, `energy_seer/tests/test_telemetry.py`
