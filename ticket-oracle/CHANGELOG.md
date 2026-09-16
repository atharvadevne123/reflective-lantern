# Changelog

All notable changes to Ticket-Oracle are documented here.

## [1.0.0] — 2026-09-16

### Added
- FastAPI REST API with 7 versioned endpoints under `/api/v1/`
- XGBoost + LightGBM + RandomForest VotingClassifier for P1-P4 priority classification
- XGBoost binary classifier for SLA breach prediction
- 6-stage sklearn ColumnTransformer pipeline (TF-IDF text, OrdinalEncoder categorical, StandardScaler numeric)
- 5-fold stratified cross-validation with weighted OvR AUC-ROC
- KS-test + PSI drift detection on prediction distributions
- SQLAlchemy ORM with PredictionLog, DriftLog, ModelMetrics tables
- Alembic-ready database schema
- Airflow weekly champion/challenger retraining DAG with AUC gate
- Docker + docker-compose with PostgreSQL
- pytest suite (90+ tests across 4 modules)
- GitHub Actions CI (ruff lint + pytest on Python 3.11/3.12 + wheel build)
- Rate limiting (slowapi), GZip middleware, correlation ID middleware
- Pydantic v2 input validation schemas
- Architecture diagram (`screenshots/architecture.png`)
