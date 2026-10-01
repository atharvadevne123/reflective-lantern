# Changelog

## [1.0.0] - 2026-09-30

### Added
- FastAPI REST API with 7 versioned `/api/v1/` endpoints
- XGBoost + LightGBM + RandomForest soft-voting ensemble
- 7-stage sklearn feature pipeline (Form, Lag/Rolling, H2H, Ratios, Fatigue, DropCat, Scaler)
- 5-fold stratified cross-validation with AUC-ROC reporting
- KS-test + PSI drift detection per feature
- SQLAlchemy ORM with PredictionLog, DriftLog, PlayerPerformance, MatchRecord tables
- Alembic migration support
- Player performance scoring (goals, assists, win-rate, fatigue, injury)
- Airflow champion/challenger weekly retraining DAG
- PostgreSQL + Docker production setup
- pytest suite with 80+ tests across 5 modules
- GitHub Actions CI (ruff lint + pytest on Python 3.11/3.12)
- Rate limiting middleware (200 req/min per IP)
- Correlation ID middleware
- GZip compression middleware
- Structured JSON logging
- Architecture diagram
