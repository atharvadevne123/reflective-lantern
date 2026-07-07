# Changelog

All notable changes to Cyber-Sentinel are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
