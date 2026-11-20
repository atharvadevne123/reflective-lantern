# Changelog

All notable changes to Ops-Vision are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-26

### Added

- **Incident prediction API** — `POST /api/v1/predict` scores six telemetry
  metrics with a soft-voting ensemble (XGBoost + LightGBM + RandomForest) and
  returns an incident flag, severity band, confidence, and runbook hint.
- **Feature pipeline** — sklearn `Pipeline` expanding six raw metrics into ten
  features (`resource_pressure`, `latency_err_ratio`, `throughput_pressure`,
  `log_latency_p99`) followed by `RobustScaler`.
- **Drift monitoring** — `DriftMonitor` runs two-sample KS tests per feature
  against a sliding reference window, flagging drift at p < 0.05.
- **Runbook retrieval** — FAISS inner-product index over TF-embedded runbooks,
  with a brute-force cosine fallback when `faiss-cpu` is unavailable.
- **Incident rate forecasting** — Holt double exponential smoothing producing a
  24-hour horizon with 80% prediction intervals.
- **Retraining DAG** — nightly Airflow DAG with a 0.70 AUC-ROC promotion gate;
  task functions remain callable when Airflow is absent.
- **Persistence** — SQLAlchemy models for `incidents`, `predictions`, and
  `drift_alerts` with pooled PostgreSQL access and indexed timestamps.
- **Middleware** — correlation-ID propagation and per-IP token-bucket rate
  limiting, exempting health and docs routes.
- **Operational endpoints** — `/health`, `/version`, `/api/v1/health`,
  `/api/v1/metrics`, `/api/v1/drift/status`.
- **166-test suite** covering API contracts, validation boundaries, feature
  correctness, model persistence, drift detection, forecasting, retrieval, and
  CRUD aggregates.
- **CI** — GitHub Actions running ruff, pytest on Python 3.10 and 3.11 with
  coverage, mypy, and a Docker build.
- **Containerisation** — multi-stage-friendly Dockerfile running as a non-root
  user with a healthcheck, plus docker-compose with PostgreSQL 16.

### Design notes

- The database engine is constructed lazily rather than at import time, so
  importing `app.database` requires neither a driver nor a reachable server.
  This is what allows the suite and CI to run without a Postgres service.
- `latency_err_ratio` clamps its denominator to 0.001. Error rate is
  legitimately zero on healthy services, and an unguarded division would emit
  `inf` and corrupt the fitted scaler.
- Engine keyword arguments are dialect-aware: SQLite rejects the QueuePool
  sizing options that PostgreSQL requires.

[1.0.0]: https://github.com/atharvadevne123/reflective-lantern/releases/tag/ops-vision-v1.0.0
