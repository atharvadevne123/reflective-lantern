# Property-Sage — Innovation Report
**Date:** 2026-09-10  
**Mode:** Automated INNOVATION MODE  

## Project Summary

Property-Sage is a production-quality real estate property valuation and rental yield prediction platform. It uses an XGBoost + LightGBM + RandomForest VotingRegressor ensemble wrapped in a sklearn Pipeline with 11 engineered features.

## Tech Stack
- **API**: FastAPI with correlation-ID middleware, rate limiting (60 req/60s), and versioned endpoints
- **ML**: XGBoost + LightGBM + RandomForest ensemble (weights 0.4/0.4/0.2)
- **Features**: 11 engineered (property age, age², sqft/bed, bath/bed, lot density, encodings)
- **RAG**: TF-IDF + cosine similarity over 10 neighbourhood market reports
- **Monitoring**: KS-test drift detection on 24h rolling window; auto-retrain trigger
- **DB**: SQLAlchemy ORM (Prediction, ModelMetrics, DriftLog); SQLite dev / PostgreSQL prod
- **Migrations**: Alembic with initial schema revision
- **Experiment tracking**: File-based JSONL run log with best-run queries
- **CI**: GitHub Actions (ruff + pytest) on push/PR
- **Containers**: Docker Compose with health-checks

## API Endpoints
- `GET /api/v1/health` — liveness
- `GET /api/v1/health/deep` — deep health with DB + model checks
- `POST /api/v1/predict` — property valuation + rental yield
- `GET /api/v1/metrics` — model performance + prediction stats
- `POST /api/v1/drift-check` — KS-test drift report
- `GET /api/v1/neighbourhood/{name}` — RAG market report
- `GET /api/v1/neighbourhood-search` — semantic neighbourhood search
- `POST /api/v1/retrain` — authenticated model retraining
- `GET /api/v1/retrain/history` — retrain audit log

## Test Results
- **98 tests**, all passing
- Coverage: features, model, monitoring, drift, RAG, health, predict, retrain endpoints

## Bugs Found & Fixed
1. `float(None)` TypeError when `lot_size` is `None` (optional field in API) — fixed with `or`-default guard in `log_prediction`.
2. Drift detection parametrized test false failure — mean shift of 2 units with std=5 still crossed p=0.05 at n=300; corrected to use identical distributions for the False case.

## Notes
- Standalone repo creation blocked (403); built inside `reflective-lantern` on branch `property-sage`.
- SMTP ports blocked by network policy; report committed to `history/reports/` instead of emailed.
