# Changelog

## [1.0.0] - 2026-09-23

### Added
- FastAPI service with `/api/v1/predict`, `/api/v1/predict/batch`, `/api/v1/health`, `/api/v1/metrics`, `/api/v1/drift`, `/api/v1/retrain` endpoints
- LightGBM ensemble: category classifier, SLA breach predictor, resolution hours estimator
- NLP feature engineering: TF-IDF (200 features), structured metadata, urgency scoring
- KS-test drift detection on 4 key feature distributions
- SQLAlchemy ORM with SQLite (dev) and PostgreSQL (prod) support
- Docker + docker-compose with PostgreSQL service
- Automated retraining pipeline (standalone + Airflow DAG)
- pytest suite with 30+ tests covering API, model, features, monitoring
- GitHub Actions CI (ruff lint + pytest)
- Correlation ID middleware for request tracing
- Pydantic v2 input validation on all endpoints
- API versioning under `/api/v1/`

### Additional features
- FAISS-based similar ticket retrieval with numpy fallback
- Token-bucket rate limiting middleware (60 req/min default)
- After-hours and weekend time-series features
- Volume anomaly detection
- MLflow experiment tracking stub
- AWS/boto3 S3 artifact upload stub
- Alembic database migrations
- Structured JSON logging configuration
- Standalone training/evaluation script
