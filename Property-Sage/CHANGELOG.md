# Changelog

All notable changes to Property-Sage are documented here.

## [1.0.0] - 2026-09-10

### Added
- FastAPI application with `/api/v1/predict`, `/api/v1/health`, `/api/v1/metrics`, `/api/v1/drift-check`
- XGBoost + LightGBM + RandomForest ensemble for property price and rental yield prediction
- sklearn Pipeline with 11 engineered features (age², sqft/bed ratio, lot density, neighbourhood index)
- SQLAlchemy ORM with SQLite (dev) and PostgreSQL (prod) support
- KS-test drift detection on 24-hour rolling prediction window
- Automated retraining pipeline triggered on statistical drift (p < 0.05)
- Docker + docker-compose deployment with PostgreSQL service
- GitHub Actions CI: ruff lint + pytest on every push
- Correlation ID middleware and structured request logging
- Pydantic v2 input validation on all endpoints
- 5-fold cross-validation with R², RMSE, MAE metrics logged to JSON
- pytest suite with parametrized tests and mocked DB sessions

## [Unreleased]
