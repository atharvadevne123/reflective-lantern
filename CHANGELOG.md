# Changelog

## [1.0.0] - 2026-07-10

### Added
- FastAPI REST API with `/api/v1/predict`, `/api/v1/anomaly`, `/api/v1/drift`, `/api/v1/metrics`, `/api/v1/health`, `/api/v1/train`
- XGBoost + LightGBM + RandomForest `VotingRegressor` ensemble for energy consumption forecasting
- 7-stage sklearn feature engineering pipeline: temporal cyclic encoding, lag features (1h–168h), rolling stats (3h/6h/24h), weather features, occupancy features, drop-non-numeric, StandardScaler
- IsolationForest anomaly detector with severity classification
- KS-test drift detection with configurable reference window
- SQLAlchemy ORM: `EnergyReading`, `PredictionLog`, `AnomalyLog`, `DriftLog`
- PostgreSQL + Docker Compose production setup
- Airflow weekly retraining DAG with R2≥0.70 and row-count≥500 gates
- Rate limiting middleware (200 req/min per IP)
- Correlation ID middleware for distributed tracing
- pytest suite with 50+ tests and parametrized cases
- GitHub Actions CI (ruff lint + pytest)
- Architecture diagram

### Stats
- Tests:  passing
- Files: 33 Python files
- Features: 7-stage feature pipeline, 30+ engineered features
