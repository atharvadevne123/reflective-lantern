# Changelog

All notable changes to Risk-Sense are documented here.

## [1.0.0] - 2026-11-11

### Added
- FastAPI service with `/api/v1/score`, `/health`, `/metrics`, `/drift`, `/version` endpoints
- XGBoost + LightGBM + Random Forest ensemble with isotonic calibration
- 5-stage sklearn feature pipeline: LogIncome, Interactions, ClipOutliers, StandardScaler
- KS-test and PSI drift detection
- Rolling default rate monitoring
- Automated retraining pipeline (drift-triggered and scheduled)
- PostgreSQL persistence for predictions and drift logs
- Comprehensive test suite: 60+ test cases across API, model, features, and monitoring
- Docker + docker-compose deployment
- Pydantic v2 request validation
- GitHub Actions CI
