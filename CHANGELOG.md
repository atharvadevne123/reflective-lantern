# Changelog

## [1.0.0] - 2026-07-10

### Added
- Initial Volt-Cast release
- XGBoost + LightGBM + RandomForest VotingRegressor ensemble
- 6-stage sklearn feature engineering pipeline
- 7 FastAPI endpoints: /predict, /batch-predict, /forecast, /drift, /retrain, /health, /metrics
- KS-test drift detection with PredictionTracker
- SQLAlchemy ORM with EnergyReading, PredictionLog, DriftReport, RetrainingLog models
- PostgreSQL + docker-compose production setup
- Airflow weekly retraining DAG with R² gate
- Rate limiting (200 req/min) + correlation ID middleware
- pytest suite with 60+ tests across 5 test modules
- GitHub Actions CI with ruff lint + pytest
