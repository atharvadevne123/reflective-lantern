# Changelog

## [1.0.0] - 2026-09-09

### Added
- FastAPI service with `/predict`, `/health`, `/metrics`, `/version`, `/drift-check` endpoints
- XGBoost + LightGBM + RandomForest soft-voting ensemble
- 5-factor sklearn feature pipeline (volatility, momentum, volume ratio, correlation, beta)
- FAISS-based historical market regime pattern search
- KS-test drift monitoring with PostgreSQL event logging
- Airflow-compatible weekly retraining DAG
- 5-fold stratified CV with AUC-ROC evaluation
- Docker + docker-compose deployment
- Pydantic v2 request/response validation
- GitHub Actions CI (lint + test)
