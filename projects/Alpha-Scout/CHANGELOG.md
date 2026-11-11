# Changelog

## [1.0.0] - 2026-11-11

### Added
- `/api/v1/rank` endpoint for scoring and ranking a portfolio of securities
- `/api/v1/feature-importance` endpoint exposing top XGBoost feature importances
- `/api/v1/anomaly` endpoint for Isolation Forest outlier detection
- `/api/v1/pipeline-health` endpoint for ML pipeline diagnostics
- Rate limiting middleware (60 req/60s per IP)
- MLflow experiment tracking stub
- AWS S3 model artefact storage stub
- Alembic database migration setup
- pydantic-settings configuration module
- Isolation Forest anomaly detection service
- Pipeline health check in monitoring module
- FastAPI application with `/api/v1/predict`, `/api/v1/health`, `/api/v1/metrics`, `/api/v1/drift` endpoints
- `StockFeatureEngineer` computing 20+ technical/fundamental features (RSI, MACD, rolling stats, momentum lags)
- XGBoost ensemble model with sklearn Pipeline and 5-fold cross-validation AUC scoring
- KS-test drift detection via `compute_drift()` with per-feature monitoring
- SQLAlchemy ORM models: `PredictionLog` and `DriftLog` (SQLite dev / PostgreSQL prod)
- Docker + docker-compose with PostgreSQL service and health checks
- Automated weekly retraining DAG (Airflow-compatible) with AUC validation gate
- Pydantic v2 request/response schemas with field validation
- Correlation ID middleware and structured logging
- pytest suite with 25+ tests: conftest, test_api, test_model, test_features, test_monitoring
- GitHub Actions CI: ruff lint + pytest
- Architecture diagram (SVG)
- `.env.example`, `Makefile`, `pyproject.toml`, `.pre-commit-config.yaml`
