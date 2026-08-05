# Changelog

## [1.0.0] - 2026-08-05

### Added
- LightGBM + XGBoost + RandomForest VotingClassifier ensemble for purchase intent prediction
- 5-stage sklearn feature engineering pipeline (ratio, interaction, lag/rolling, discount encoding, scaling)
- FAISS item-similarity nearest-neighbour index with brute-force fallback
- FastAPI app with 5 versioned endpoints under `/api/v1/`
- KS-test drift detection with rolling reference window
- SQLAlchemy ORM with PredictionLog, DriftLog, UserProfile, ItemCatalog tables
- Docker + docker-compose with PostgreSQL
- Airflow champion/challenger weekly retraining DAG
- pytest suite with 60+ tests across 4 modules
- GitHub Actions CI (ruff lint + format + pytest + wheel build)
- Pydantic v2 input validation on all endpoints
- Rate limiting middleware (200 req/min per IP)
- Correlation-ID and GZip middleware
- Structured JSON logging throughout
- `.env.example` with all required environment variables
