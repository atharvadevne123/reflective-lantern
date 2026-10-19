# Changelog

All notable changes to Social-Pulse are documented here.

## [1.0.0] — 2026-10-14

### Added
- FastAPI application with `/predict`, `/health`, `/metrics`, `/drift`, `/train`, `/version` endpoints
- LightGBM + XGBoost + RandomForest + LogisticRegression VotingClassifier ensemble
- NLP feature pipeline: 17 hand-crafted stats + TF-IDF bigrams (200 features)
- Cyclical hour-of-day and day-of-week encoding
- Follower count log-scale bucketing (7 buckets)
- SQLAlchemy models: `predictions`, `drift_records`, `training_runs`
- KS-test drift detection across 6 key features
- Airflow DAG for weekly drift-triggered retraining
- 5-fold stratified cross-validation with AUC-ROC scoring
- Correlation ID middleware for request tracing
- Pydantic input validation with field-level constraints
- Docker and docker-compose setup with PostgreSQL
- pytest test suite: 60+ tests across API, model, features, and monitoring
- GitHub Actions CI: lint + test
- Architecture diagram
