# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

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
- `app/tariff.py` — flat, time-of-use, and tiered electricity pricing with scheme comparison
- `app/load_profile.py` — base load, load factor, peak-to-average, ramp rate, profile class
- `app/weather_normalization.py` — degree-day adjustment separating weather from efficiency change
- `app/demand_response.py` — customer baseline load, curtailment measurement, event settlement
- `app/power_quality.py` — power factor, reactive power, voltage imbalance, capacitor sizing
- `app/solar.py` — PV generation, hourly self-consumption split, payback with degradation
- `app/battery.py` — storage dispatch simulation, peak shaving, capacity sizing
- `app/energy_benchmark.py` — peer-cohort EUI ranking with 1-100 score and letter grade
- 11 API endpoints exposing the above under `/api/v1/`
- `docs/energy_analytics.md` — full reference for the energy analytics modules
- 397 tests across the new modules, endpoints, and expanded existing suites

### Fixed
- Root logger state is now restored between tests. `configure_logging` clears the
  root logger's handlers, which removed pytest's capture handler and silently
  broke `caplog` for every test that ran afterwards.
- Two test classes in `tests/test_event_bus.py` shadowed earlier definitions of
  the same name, silently dropping the shadowed tests from the run.

- `app/data_augmentation.py` — text and numeric training data augmentation
- `app/correlation_id.py` — thread-local request tracing context
- `app/health_check.py` — composable readiness/liveness probe registry
- `app/metrics_collector.py` — counters, gauges, and histograms
- `app/task_queue.py` — priority task scheduling with worker threads
- `app/config_validator.py` — schema-based configuration validation
- `app/audit_log.py` — immutable append-only structured audit trail
- `app/notification_dispatcher.py` — multi-channel severity-routed notifications
- `app/webhook_handler.py` — HMAC-verified inbound webhook processing
- `app/retry.py` — exponential backoff decorator
- `app/circuit_breaker.py` — CLOSED/OPEN/HALF_OPEN state machine
- `app/pagination.py` — offset and cursor-based pagination helpers
- `app/event_bus.py` — synchronous pub/sub with wildcard subscriptions
- `app/profiler.py` — wall-clock timing and call-stats decorators
- `app/token_bucket.py` — thread-safe rate limiter
- `app/alerting.py` — threshold-based alert rules with cooldown
- `app/experiment_tracker.py` — deterministic A/B variant assignment
- `app/feature_store.py` — versioned feature set storage
- `app/model_registry.py` — ML model lifecycle management
- `app/data_versioning.py` — SHA-256 checksummed data lineage
- `app/batch_processor.py` — chunked batch execution with callbacks
- `app/geo_utils.py` — haversine distance, bounding boxes, nearest neighbour
- `app/compression.py` — zlib/gzip compress/decompress with JSON helpers
- `app/cost_estimator.py` — USD cost projection from resource specs
- `app/shadow_mode.py` — parallel shadow traffic comparison
- Comprehensive pytest suites for every module above
- `docs/architecture.md` — layered architecture overview
- `docs/api_reference.md` — public API surface documentation
- `docs/deployment.md` — environment setup and deployment guide
- `docs/monitoring.md` — observability runbook
- `scripts/benchmark.py` — micro-benchmark suite
- `scripts/seed_data.py` — development data seeding

---

## [1.0.0] — Initial release

### Added
- Initial project scaffold with `app/config.py`
- Basic CI workflow
