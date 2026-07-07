# Changelog

All notable changes to Cyber-Sentinel are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.1] - 2026-07-07

### Fixed
- `get_feature_importance` crashed on fitted `VotingClassifier` (`estimators_` holds bare estimators; now uses `named_estimators_`)
- `/predict`, `/metrics`, and `/feature-importance` now return 503 (not 500) when the model is untrained (`ModelNotTrainedError` was never caught by `except RuntimeError`)
- `train_model` now persists `metrics.pkl` so `load_metrics()` serves real data
- Test suite uses a SQLite database (set before app import) instead of requiring PostgreSQL

### Added
- Cache hit/miss counters exposed via `/cache/stats`
- `DRIFT_THRESHOLD` environment variable to tune KS-test sensitivity
- `clear_index_cache()` for attack-pattern index lifecycle management
- `FEATURE_NAMES` aligned 1:1 with the 25-element feature vector
- Makefile `coverage` and `check` targets

## [1.0.0] - 2026-06-29

### Added
- FastAPI REST API with `/predict`, `/train`, `/metrics`, `/drift`, `/health`, `/version`, `/feature-importance` endpoints
- XGBoost + LightGBM + RandomForest soft-voting ensemble classifier
- KS-test drift detection via `scipy.stats.ks_2samp`
- FAISS-based attack pattern matching with numpy fallback
- SQLAlchemy ORM models: `NetworkEvent`, `Prediction`, `DriftLog`
- Airflow retraining DAG with weekly schedule
- Docker and docker-compose configuration with PostgreSQL
- Alembic migrations setup
- Full pytest suite with >50% coverage
- GitHub Actions CI with lint, type-check, and test stages
- Pre-commit hooks (ruff, trailing-whitespace)
- Correlation ID middleware for request traceability
- Input validation via Pydantic v2 models
- Feature engineering pipeline with 25 network traffic features
