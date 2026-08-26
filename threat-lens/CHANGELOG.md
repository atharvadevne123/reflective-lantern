# Changelog

All notable changes to Threat-Lens are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-08-26

### Added

- FastAPI service exposing five versioned endpoints under `/api/v1`:
  `health`, `predict`, `metrics`, `drift`, and `threats`.
- Soft-voting ensemble (XGBoost + LightGBM + RandomForest) behind an sklearn
  `Pipeline`, classifying flows as `normal`, `dos`, `probe`, `r2l`, or `u2r`.
- Feature pipeline deriving 28 features from 23 raw flow fields, including
  `bytes_ratio`, `bytes_per_second`, `error_rate_combined`, `connection_density`,
  and a high-risk-service encoding.
- Threat-intelligence retriever over a CVE / MITRE ATT&CK corpus, attaching
  context to every flow classified as an attack.
- KS-test drift monitoring with per-feature reports persisted to `drift_reports`.
- Prediction logging to `prediction_logs` with correlation IDs.
- Airflow DAG for nightly retraining, plus a scheduler-free
  `run_retraining_pipeline()` entry point.
- Pydantic v2 request validation; unknown protocols rejected with HTTP 422.
- Correlation-ID and response-timing middleware.
- Docker image and `docker-compose.yml` wiring the API to PostgreSQL.
- 50-test pytest suite covering API, model, features, and monitoring.
- GitHub Actions CI running ruff and pytest.

### Fixed

- `cross_val_score(n_jobs=-1)` nested against `RandomForestClassifier(n_jobs=-1)`
  deadlocked joblib on constrained runners; base estimators are now
  single-threaded so only the CV loop parallelises.
- `NetworkFeatureEngineer.transform([])` returned a 1-D empty array, surfacing
  downstream as an opaque "Expected 2D array" error; it now returns a correctly
  shaped `(0, n_features)` matrix.
- `generate_synthetic_dataset` produced an empty dataset for `n_samples < 5`;
  it now emits at least one row per class.

[1.0.0]: https://github.com/atharvadevne123/reflective-lantern/releases/tag/threat-lens-v1.0.0
