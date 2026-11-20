# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-10-09

### Added (2026-09-07 — Reflective Lantern improvement run)
- Google-style docstrings added to all previously undocumented methods across
  `app/metrics_collector.py`, `app/profiler.py`, `app/task_queue.py`,
  `app/circuit_breaker.py`, `app/cost_estimator.py`, `app/features.py`,
  `app/health_check.py`, `app/token_bucket.py`, `app/audit_log.py`,
  `app/batch_processor.py`, `app/config_validator.py`, `app/experiment_tracker.py`,
  `app/faiss_index.py`, `app/geo_utils.py`, `app/logging_config.py`,
  `app/middleware.py`, `app/model_registry.py`, `app/notification_dispatcher.py`,
  `app/retry.py`, and `app/alerting.py`
- `pytest.mark.parametrize` test classes added to all test modules that lacked them:
  `test_audit_log.py`, `test_correlation_id.py`, `test_notification_dispatcher.py`,
  `test_task_queue.py`, `test_middleware.py`, `test_health_check.py`,
  `test_config_validator.py`, `test_data_augmentation.py`, `test_event_bus.py`,
  `test_metrics_collector.py`, `test_webhook_handler.py`, `test_foundry_client.py`,
  `test_integration.py`, `test_foundry_export.py`, `test_foundry_sync.py`,
  `test_api_analytics.py`, and `test_alerting.py`
- CLI docstrings to `scripts/cleanup.py`, `scripts/validate_history.py`,
  `scripts/summarize_history.py`, `scripts/report_generator.py`,
  `scripts/run_all_checks.py`, `scripts/seed_data.py`, and `scripts/benchmark.py`
- `__all__` export lists to `scripts/benchmark.py`, `scripts/lantern_env.py`,
  `scripts/seed_data.py`, `scripts/check_ci_status.py`, `scripts/send_report.py`,
  and `scripts/gen_report_raw.py`

### Added

- `app/fuel_cell.py`: hydrogen fuel cell modelling (Nernst voltage, overpotentials, stack power, H2 consumption)
- `app/tidal_power.py`: tidal stream and barrage energy calculations
- `app/geothermal.py`: geothermal resource and plant modelling with LCOE and gradient functions
- `app/peak_shaving.py`: battery peak-shaving controller with revenue and effectiveness helpers
- `app/cogeneration.py`: CHP efficiency, CO2 savings, heat-to-power ratio, and PES factor
- `app/energy_storage_degradation.py`: battery degradation modelling (cycle, calendar, SoH, throughput)
- `app/renewable_curtailment.py`: curtailment analysis with carbon abatement and forecast accuracy
- `app/distributed_energy.py`: DER portfolio management with capacity factor and net metering
- `app/energy_intensity.py`: EUI benchmarking with sector rankings and carbon calculations
- `app/grid_balancing.py`: grid frequency regulation, droop response, inertia, and frequency nadir
- Extended test suites for all ten new modules with parametrized edge-case coverage
- Expanded edge-case tests for `ev_charging`, `power_quality`, `voltage_drop`, `energy_audit`,
  `flexibility_score`, `tariff`, `roi_calculator`, `microGrid_controller`, `load_profile`,
  `grid_stability`, `water_heating`, `thermal_comfort`, `occupancy_model`, `wind_power`,
  `solar_pv`, `battery_storage`, `heat_pump`, `insulation_calculator`, `carbon_budget`,
  and `demand_forecaster`
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
