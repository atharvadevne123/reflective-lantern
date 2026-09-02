# Changelog

All notable changes to Cyber-Guard are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-09-02

### Added

- FastAPI service with four versioned endpoints: `/api/v1/predict`,
  `/api/v1/health`, `/api/v1/metrics`, `/api/v1/drift`.
- XGBoost + RandomForest soft-voting ensemble classifying connections into
  five threat classes (`normal`, `dos`, `probe`, `r2l`, `u2r`).
- Fifteen-feature engineering pipeline: categorical encodings, byte ratios,
  log transforms, interaction terms, and 5-connection rolling statistics.
- KS-test drift detection over the `src_bytes` distribution with configurable
  p-value threshold and reference window.
- SQLAlchemy persistence of predictions and drift checks (SQLite for
  development, PostgreSQL for production).
- Airflow DAG for weekly retraining, gated on ≥ 70% cross-validated accuracy.
- Correlation-ID and response-time middleware on every request.
- Docker and docker-compose deployment with a health-checked PostgreSQL service.
- Test suite of 40 tests covering the API, model, features, and monitoring.
- GitHub Actions CI running ruff lint followed by pytest.

### Fixed

- Test isolation: `log_prediction` commits, so the `db_session` fixture now
  truncates all tables on teardown instead of relying on a rollback.
- Removed the deprecated `use_label_encoder` parameter from `XGBClassifier`,
  which XGBoost 2.x ignores and warns about.

[1.0.0]: https://github.com/atharvadevne123/reflective-lantern/releases/tag/cyber-guard-v1.0.0
