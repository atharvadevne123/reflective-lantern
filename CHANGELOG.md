# Changelog

## [1.0.0] - 2026-07-06

### Added
- XGBoost + LightGBM + RandomForest VotingRegressor ensemble
- 5-stage sklearn feature engineering pipeline
- FAISS comparable property search
- KS-test drift detection per feature
- FastAPI with 7 versioned `/api/v1/` endpoints
- Airflow weekly retraining DAG with R2 gate
- SQLAlchemy ORM with PostgreSQL/SQLite support
- Rate limiting (300 req/min) and correlation ID middleware
- Docker + docker-compose production setup
- GitHub Actions CI with ruff + pytest
- 5-fold cross-validation with R2 and RMSE metrics
