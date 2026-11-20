# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-08-05

### Added

**Model and features**
- LightGBM + XGBoost + RandomForest soft-voting ensemble for purchase intent
- 5-stage sklearn feature pipeline: ratios, interactions, lag/rolling, discount
  encoding, scaling — 29 features from 16 raw columns
- Transformers fit inside CV folds, so reported AUC carries no transform leakage
- `make_purchase_labels` generates signal-bearing synthetic labels from a latent
  propensity score, giving a meaningful 0.717 CV AUC instead of near-chance
- FAISS flat-L2 item-similarity index with an exact brute-force fallback

**API**
- `POST /api/v1/predict` — single user-item intent scoring
- `POST /api/v1/predict/batch` — up to 500 pairs in one vectorised pass
- `POST /api/v1/recommend` — candidate generation + intent-ranked top-K
- `POST /api/v1/similar` — FAISS nearest-neighbour item lookup
- `POST /api/v1/drift` — KS + PSI drift analysis
- `GET /api/v1/health`, `/metrics`, `/model/info`, `/cache/stats`
- Pydantic v2 validation, OpenAPI summaries and descriptions on every endpoint
- Correlation-ID, rate-limiting (200 req/min per IP), and GZip middleware

**Monitoring**
- KS-test drift detection over a bounded rolling reference window
- PSI as a sample-size-stable effect size alongside the KS p-value, graded
  stable / moderate / major, with empty bins floored to keep the score finite
- Cross-field payload coherence checks that downgrade confidence rather than reject
- Thread-safe TTL cache (300s, LRU past 1000 entries) for recommendations and similarity

**Infrastructure**
- SQLAlchemy ORM with four tables; Alembic migration for the initial schema
- Docker image running as a non-root user with a readiness-gated healthcheck
- docker-compose with PostgreSQL and a startup health gate
- Airflow champion/challenger weekly retraining DAG with AUC and row-count gates
- GitHub Actions CI: ruff lint, format check, pytest, wheel build on 3.11 and 3.12
- 180+ tests across nine modules

### Fixed

- **Champion/challenger comparison was against itself.** `train_model` writes
  `metrics.json` as a side effect, and `retrain_task` read the champion's AUC from
  that file *after* training — loading the challenger's own metrics. The gate was
  `auc >= auc`, always true, so every challenger was promoted including outright
  regressions. The champion is now read before training, and a rejected run
  restores the champion's metrics.
- **Ad-hoc index builds clobbered the serving FAISS index.** `build_faiss_index`
  wrote to the shared index path unconditionally, so any caller building a
  throwaway index replaced the deployed one — leaving a mismatched-dimension index
  that failed inside FAISS on an opaque assertion. Persistence is now opt-in,
  `load_faiss_index` rejects a dimension mismatch and rebuilds, and
  `search_similar_items` raises a clear error instead of letting the library assert.
