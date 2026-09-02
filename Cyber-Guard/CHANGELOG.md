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
- IsolationForest anomaly detection with a dedicated `/api/v1/anomaly`
  endpoint, for traffic matching none of the five known classes.
- Sliding-window rate limiting (`429` + `Retry-After`), with `/health` exempt
  so a liveness probe cannot rate-limit the service out.
- Centralised settings in `app/config.py`, read from the environment with
  safe fallbacks on malformed values.
- Weighted one-vs-rest AUC-ROC alongside accuracy in the CV report.
- Optional MLflow tracking and optional S3 artifact storage, both degrading
  to no-ops when unconfigured.
- Alembic migrations wired to the application's own metadata and
  `DATABASE_URL`.
- `scripts/seed_data.py` to backfill both drift windows so the KS path can be
  exercised on a fresh deployment.
- Test suite of 100 tests covering the API, model, features, monitoring,
  anomaly detection, rate limiting, configuration, retraining, and
  train/serve parity.
- GitHub Actions CI running ruff lint followed by pytest.

### Fixed

- Test isolation: `log_prediction` commits, so the `db_session` fixture now
  truncates all tables on teardown instead of relying on a rollback.
- Removed the deprecated `use_label_encoder` parameter from `XGBClassifier`,
  which XGBoost 2.x ignores and warns about.
- **Train/serve skew in the rolling window features.** The model is fitted on
  batches but served one connection at a time, and a rolling window over a
  one-row frame collapses to `mean == src_bytes`, `std == 0`. That is a
  systematically wrong value rather than a noisy one, and it put every served
  request off the training manifold — the anomaly detector flagged 100% of
  traffic. The engineer now learns those columns' training averages at `fit`
  time and imputes them when the frame is shorter than the window.
  Single-row and batch anomaly rates now agree (3.0% vs 3.0%).
- **Synthetic data carried no learnable signal.** Labels were drawn
  independently of the features, capping AUC at chance (measured 0.490). The
  generator now samples packet fields conditional on the threat class, giving
  AUC 0.997 and accuracy 0.960.
- **`metrics.json` could contain bare `NaN`.** A CV fold containing none of a
  rare class yields a NaN AUC; sklearn warns rather than raising, so the
  value reached `json.dump` and produced output that strict JSON parsers
  reject. Folds are now averaged with `nanmean`, an all-NaN result becomes
  `null`, and `allow_nan=False` makes any regression fail loudly.
- **`python scripts/seed_data.py` could not import `app`.** Running a file in
  `scripts/` puts that directory on `sys.path` rather than the project root;
  the script now prepends the root itself.

[1.0.0]: https://github.com/atharvadevne123/reflective-lantern/releases/tag/cyber-guard-v1.0.0
