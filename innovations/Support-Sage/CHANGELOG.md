# Changelog

## [1.0.0] - 2026-11-25

### Added
- Initial release of Support-Sage
- FastAPI endpoints: `/api/v1/predict`, `/api/v1/health`, `/api/v1/metrics`, `/api/v1/train`
- XGBoost + LightGBM + RandomForest soft-voting ensemble for priority classification
- LightGBM regressor for resolution time estimation
- TF-IDF + meta-feature engineering pipeline (5000 TF-IDF features + 24 meta features)
- FAISS-based similar ticket retrieval (RAG)
- KS-test drift detection on confidence scores, resolution times, and word counts
- SQLAlchemy models for prediction logging (SQLite dev / PostgreSQL prod)
- Docker + docker-compose deployment with PostgreSQL
- Airflow DAG for weekly automated retraining
- pytest suite with 30+ tests across API, model, features, and monitoring
- GitHub Actions CI (ruff lint + pytest)
- Architecture diagram
