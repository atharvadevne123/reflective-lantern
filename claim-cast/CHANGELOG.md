# Changelog

## [1.0.0] - 2026-11-18
### Added
- FastAPI app with `/api/v1/predict`, `/api/v1/health`, `/api/v1/metrics`, `/api/v1/drift`, `/api/v1/premium`, `/api/v1/triage/{priority}` endpoints
- XGBoost ensemble classifier with 5-fold CV for claim severity prediction
- Isolation Forest anomaly detection for outlier claims
- Actuarial feature engineering pipeline (12 features: severity ratio, claim age tiers, frequency flags, recency, geo risk)
- KS-test drift detection with SQLAlchemy persistence
- Automated weekly retraining DAG (Airflow-compatible)
- Docker + docker-compose with PostgreSQL
- GitHub Actions CI (ruff lint + pytest)
- Pydantic v2 request/response schemas with field validation
- Structlog correlation ID middleware
- Rate limiting middleware (60 req/min)
- Risk-adjusted premium recommendation endpoint
- Triage priority SLA lookup endpoint

## [1.0.1] - 2026-11-18
### Added
- Batch prediction endpoint (POST /api/v1/predict/batch)
- Claim statistics endpoint (GET /api/v1/claims/stats)
- Feature importance endpoint (GET /api/v1/feature-importance)
- Retrain trigger endpoint (POST /api/v1/retrain)
- MLflow experiment tracking stub
- AWS S3 artifact storage stub
- Alembic migration configuration
- Synthetic data generator (app/data.py)
- Input validation utilities (app/validators.py)
- Cross-validation function with StratifiedKFold
- Drift summary function in monitoring module
- bulk_claims pytest fixture
- 20+ additional tests (now 75+ total)
