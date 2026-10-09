# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-10-09

### Added

- `peak_curtailment_hour` in `demand_response` — index of maximum curtailment hour
- `below_median_cohort` in `energy_benchmark` — cohort comparison against median EUI
- `match_rate` method on `ShadowRunner` in `shadow_mode` — fraction of shadow calls matching primary
- `is_full` method on `PerKeyTokenBucket` in `token_bucket` — capacity check per key
- `has_handler` method on `WebhookHandler` in `webhook_handler` — handler registration query
- `is_empty` and `priority_range` methods on `TaskQueue` in `task_queue`
- `records_kwh_average` in `energy_export` — average kWh across energy records
- `degree_day_ratio` in `weather_normalization` — ratio of current to baseline degree-days
- `load_unbalance_factor` in `power_quality` — maximum load deviation as percentage
- `alerts_by_source` in `notifications` — group alerts by source attribute
- `disabled_channels` method on `NotificationDispatcher` in `notification_dispatcher`
- `registered_model_names` method on `ModelRegistry` in `model_registry`
- `optional_field_names` in `config_validator` — sorted list of optional schema fields
- `clamp_similarity` in `similarity` — clamp value to [-1, 1]
- `consumption_delta` in `reporting` — signed kWh difference
- `has_duplicates` in `data_quality` — non-None duplicate detection
- `pipeline_step_count` in `pipeline_utils` — count of pipeline steps
- `region_ids_with_minimum_load` in `regions` — filter regions by minimum load
- `max_absolute_error` in `metrics` — maximum |actual - predicted|
- `listing_age_bucket` in `market_context` — DOM-based age bucket label
- `arithmetic_midpoint` in `geo_utils` — arithmetic midpoint of two lat/lon pairs
- `sign_changes` in `stats_utils` — count of zero-crossings in a sequence
- `cumulative_sum` in `time_series` — running total of a sequence
- `peak_hour_index` in `load_profile` — index of maximum load hour
- `cost_per_hour_average` in `tariff` — average hourly energy cost
- `panel_efficiency_ratio` in `solar` — actual-to-rated output ratio
- `is_profitable` in `investment` — profitability check
- `co2_intensity_label` in `carbon` — human-readable grid intensity classification
- `end_of_month` and `elapsed_seconds` helpers to `app/date_utils`
- `haversine_distance` tuple-based convenience wrapper to `app/geo_utils`
- Parametrized test coverage for `test_api_analytics`, `test_foundry_client`,
  `test_foundry_export`, `test_foundry_sync`, `test_integration`, and
  `test_date_utils`

### Changed

- Upgraded `actions/upload-artifact` from v7 to v4 in CI workflow
- Applied ruff format and lint fixes across all Python source files
- Expanded `.env.example` with security, performance, and rate-limit variables
- Added `secret_key`, `max_workers`, `request_timeout_s`, `drift_ks_threshold`,
  and `reference_buffer_size` settings to `Settings` dataclass
- Replaced all nested `with` statements with combined form (SIM117)

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
