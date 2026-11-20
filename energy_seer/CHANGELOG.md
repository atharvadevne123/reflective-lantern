# Changelog

## [1.1.0] - 2026-10-26

### Added
- `validate_temperature`, `validate_humidity`, `validate_horizon` in `validators.py`
- `weighted_moving_average` and `exponential_smoothing` forecasters
- `psi_alert` function for human-readable PSI alerts
- `summarise_anomalies` helper in `monitoring.py`
- `efficiency_percentile` in `efficiency_score.py`
- `demand_trend` function in `demand_spike.py`
- `compute_anomaly_rate` helper in `grid_report.py`
- `counter_exists` and `reset_prefix` in `telemetry.py`
- `get_with_default` and `keys` helpers in `cache.py`
- `/api/v1/version` endpoint
- `scripts/health_check.py` liveness probe
- `train_time_sec` in model training metrics
- 50+ new unit tests across all modules

### Changed
- `Makefile` — added `health`, `typecheck`, and `test-fast` targets
- `_grade()` in `efficiency_score.py` memoised with `@lru_cache`
- Module-level docstrings improved throughout

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
