# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

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
