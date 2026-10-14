# Changelog

All notable changes to Trade-Sage are documented in this file.

## [1.0.0] - 2026-10-14

### Added

**Core ML Pipeline**
- XGBoost + LightGBM + RandomForest soft-voting VotingClassifier ensemble
- 22 engineered features across three transformer classes: TechnicalIndicators, LagRollingFeatures, RatioFeatures
- StratifiedKFold(5) cross-validation with AUC-ROC champion tracking

**API**
- FastAPI service with six endpoints under `/api/v1/`: `predict`, `portfolio/risk`, `drift`, `health`, `metrics`, `train`
- Pydantic v2 input validation with cross-field model_validator
- GZip compression, Correlation ID middleware, IP-based rate limiting (200 req/min)
- Thread-safe TTL/LRU prediction cache (512 entries, 300s TTL)

**Portfolio Risk**
- Historical VaR and Expected Shortfall (CVaR) at configurable confidence levels
- Annualised Sharpe ratio and maximum drawdown
- Average pairwise correlation matrix

**Drift Monitoring**
- KS-test per feature against a 500-sample reference window
- Automatic flagging and DB logging when p < 0.05

**Database**
- SQLAlchemy ORM with lazy engine initialisation
- Tables: `prediction_logs`, `drift_logs`, `model_metrics`
- Alembic migration environment and initial schema migration

**Backtesting**
- Long-only signal backtester with transaction costs, equity curve, Sharpe, and win rate

**Observability**
- Structured JSON logging via `logging_config.py`
- Thread-safe `ServiceMetrics` counters with `snapshot()` export
- `HealthStatus` dataclass for liveness/readiness probes

**Infrastructure**
- Docker + PostgreSQL docker-compose setup with non-root user
- GitHub Actions CI matrix (Python 3.11/3.12) with ruff lint and pytest
- Airflow champion/challenger DAG stub
- MLflow and AWS S3 no-op stubs
- FAISS equity pattern similarity index with brute-force fallback

**Testing**
- 166 tests across 16 test modules
- 100% ruff lint compliance
- StaticPool in-memory SQLite for full test isolation

### Security
- Input validation at all API boundaries
- No raw SQL string formatting
- Non-root Docker container
- CORS origin configuration via environment variable
