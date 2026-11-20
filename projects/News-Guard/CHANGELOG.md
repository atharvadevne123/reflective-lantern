# Changelog

## [1.0.0] - 2026-10-28

### Added
- FastAPI service with `/api/v1/predict`, `/api/v1/health`, `/api/v1/metrics`, `/api/v1/drift`
- XGBoost + RandomForest VotingClassifier ensemble
- TF-IDF headline feature extraction (300 n-gram features)
- 13 hand-crafted linguistic features: clickbait score, caps ratio, sensational words, source credibility
- KS-drift detection comparing reference vs recent prediction distributions
- SQLAlchemy models for `PredictionLog` and `DriftLog`
- Docker + docker-compose with PostgreSQL
- Airflow DAG for weekly automated retraining
- pytest suite with 5 test files and 37 parametrized cases
- GitHub Actions CI (ruff lint + pytest)
- Architecture diagram
