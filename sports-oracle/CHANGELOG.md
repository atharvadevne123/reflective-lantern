# Changelog

## [1.0.0] - 2026-10-07

### Added
- FastAPI service with 6 versioned endpoints (`/api/v1/predict`, `/batch`, `/health`, `/metrics`, `/drift`, `/retrain`)
- XGBoost + LightGBM + RandomForest soft-voting ensemble classifier
- 5-stage sklearn feature engineering pipeline (FormIndex, H2H, AttackDefense, RestDays, DropCategorical)
- KS-test + PSI drift detection per feature column
- SQLAlchemy ORM (Match, PredictionLog, DriftLog, ModelMetrics)
- PostgreSQL + Docker + docker-compose
- Airflow champion/challenger retraining DAG with AUC ≥ 0.65 gate
- pytest suite (85+ tests across 5 modules)
- GitHub Actions CI (ruff lint + pytest 3.11/3.12 + wheel build)
- 5-fold stratified cross-validation AUC-ROC
- Rate limiting middleware (200 req/min)
- Correlation ID middleware for distributed tracing
- GZip compression middleware
- Input validation via Pydantic v2 models
- Architecture diagram
