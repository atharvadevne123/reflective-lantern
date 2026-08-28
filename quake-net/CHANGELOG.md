# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-20

Initial release.

### Added

- **Prediction API** — FastAPI service with ten versioned endpoints under `/api/v1`:
  single and batch magnitude prediction, health, metrics, KS drift, PSI drift, similarity
  search, anomaly scoring, recent events, and cache statistics.
- **Voting ensemble** — XGBoost (0.6) + RandomForest (0.4) `VotingRegressor` predicting
  local magnitude, with 5-fold cross-validated R² plus held-out RMSE / MAE / R².
- **Aftershock head** — logistic transform of predicted magnitude centred at M5.0, and a
  USGS-style magnitude class band.
- **Six-stage feature pipeline** — seismic moment proxies, S/P amplitude ratio,
  depth-corrected amplitudes, station density, fault-mechanism encoding, rolling and lag
  features over 3/5/10-event windows; 8 raw columns expand to 47 features.
- **Drift monitoring** — per-feature Kolmogorov–Smirnov tests against a stored reference
  distribution, Population Stability Index, and a bounded thread-safe prediction store.
- **Anomaly detection** — Isolation Forest over seismic signatures alongside z-score and
  Tukey IQR rules, reported side by side.
- **Similarity search** — FAISS `IndexFlatL2` over historical event signatures with an
  exact brute-force NumPy fallback when FAISS is unavailable.
- **Aftershock forecasting** — modified Omori law (`n(t) = K/(t+c)^p`) with closed-form
  interval integration, maximum-likelihood fitting of K and p over a decay-exponent grid,
  Bath's-law largest-aftershock estimate, and rate half-life, exposed at
  `POST /api/v1/forecast/aftershocks`.
- **Input validation helpers** — shared range, completeness and S/P-coherence checks used
  by both the API and batch ingestion.
- **Automated retraining** — Airflow-compatible weekly DAG with a champion/challenger gate
  requiring both an improvement over the incumbent and an absolute R² floor of 0.70.
- **Persistence** — SQLAlchemy models for seismic events, drift logs and model metrics;
  SQLite for development, PostgreSQL 16 for production.
- **Operational middleware** — correlation-ID propagation, per-IP rate limiting
  (200 req/min), GZip compression, and structured JSON access logging.
- **TTL cache** — thread-safe LRU cache with per-entry expiry and hit-rate statistics.
- **Infrastructure** — Dockerfile (non-root user, healthcheck), docker-compose with
  PostgreSQL, Makefile, pre-commit hooks, and GitHub Actions CI running ruff and pytest on
  Python 3.11 and 3.12.
- **Tests** — 293 tests across twelve modules, with transactional database isolation.

### Fixed

- `DropCategoricalColumns` selected columns via `dtype == object`, which silently missed
  string columns under the PyArrow-backed pandas default and let non-numeric data reach
  `StandardScaler`. Selection is now by "not numeric dtype".
- The champion/challenger gate read the incumbent's R² from `metrics.json` *after*
  training had already overwritten that file, comparing the challenger against itself and
  promoting every model including regressions. The champion is now read before training,
  and a rejected run restores its metrics.
- Ad-hoc similarity index builds unconditionally overwrote the persisted FAISS index that
  the API serves from. Persistence is now opt-in.
- `train_model` reported `n_features` as the raw input width (8) rather than the
  post-pipeline width the model actually sees (47), understating the feature space in
  every metrics record. Both are now reported.
- The `model_metrics` table was defined but never written to, so training history was lost
  on every retrain. `record_training_run` now appends each run, and the retraining DAG
  opts in; a persistence failure is logged without aborting the training run.
