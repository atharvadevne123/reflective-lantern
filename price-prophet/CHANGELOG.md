# Changelog

All notable changes to Price-Prophet are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-27

### Added
- FastAPI REST service with `/predict`, `/batch-predict`, `/health`, `/metrics` endpoints
- XGBoost + Linear ensemble demand forecasting pipeline
- Price elasticity estimation with confidence intervals
- Backtesting framework for strategy evaluation
- KS-test distribution drift detection and alerting
- SQLite/PostgreSQL data persistence with SQLAlchemy ORM
- Alembic database migration support
- CLI (`app/cli.py`) for training, prediction, and export
- Full pytest test suite with 20+ test modules
- Docker and docker-compose configuration
- GitHub Actions CI workflow with ruff linting and pytest
- Makefile with install, test, lint, run, and docker targets
- `.env.example` with all configurable environment variables
- `.pre-commit-config.yaml` with ruff and trailing-whitespace hooks

### Changed
- N/A (initial release)

### Fixed
- N/A (initial release)
