# Changelog

## [1.0.0] - 2026-07-14

### Added
- XGBoost + LightGBM + RandomForest VotingRegressor ensemble
- 7-stage sklearn feature engineering pipeline (lag, rolling, temporal, weather, building, drop, scaler)
- IsolationForest + Z-score anomaly detection
- KS-test drift detection per feature
- FastAPI with 8 versioned /api/v1/ endpoints
- Multi-step hourly forecast endpoint
- Batch prediction endpoint
- FAISS-based similar pattern retrieval (RAG)
- SQLAlchemy ORM with PostgreSQL/SQLite
- Airflow weekly retraining DAG with R2 gate
- Docker + docker-compose setup
- pytest suite with 60+ tests
- GitHub Actions CI (ruff lint + pytest)
- Rate limiting middleware (300 req/min per IP)
- Correlation ID middleware
- Alembic database migrations
- 5-fold cross-validation R2/RMSE metrics
- MLflow-ready model tracking hooks
