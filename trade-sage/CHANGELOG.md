# Changelog

All notable changes to Trade-Sage are documented in this file.

## [1.1.0] - 2026-10-29

### Added

**Analytics**
- `compute_calmar_ratio` in `portfolio.py` — annualised return divided by max drawdown
- `aggregate_signals` in `signals.py` — multi-signal consensus voting with mean confidence
- `summarise_backtest` and `compare_strategies` in `backtest.py` — JSON-ready summaries and side-by-side strategy comparison
- `compute_log_returns` and `add_date_features` in `preprocessing.py`

**Utilities**
- `TTLCache.stats()` — returns `{size, maxsize}` metadata
- `RequestTimingMiddleware` — adds `X-Process-Time` (ms) header to responses
- `index_size()` and `clear_index()` in `faiss_index.py`
- `is_mlflow_available()` and `log_run_safe()` in `mlflow_stub.py`
- `is_s3_configured()` and `list_model_artifacts()` in `boto3_stub.py`
- `reset_metrics()` and `record_prediction_latency()` in `metrics.py`
- `get_logger()` and `silence_noisy_loggers()` in `logging_config.py`
- `check_env_vars()` and `liveness_probe()` in `health.py`
- `reset_config()`, `config_as_dict()`, `_redact_url()` in `config.py`
- `count_predictions()` and `count_drift_alerts()` in `database.py`
- `validate_feature_count()` in `features.py`
- `get_drift_summary()` and `clear_reference_store()` in `monitoring.py`
- `importance_as_dict()` and `cumulative_importance()` in `feature_importance.py`

**Validation Schemas**
- `TrainRequest`, `TrainResponse`, `VersionResponse` Pydantic models

**Tests**
- 15 new test files covering all new utilities, edge cases, and parametrized scenarios
- Import order fixes across 7 test files

### Changed
- `TTLCache` docstrings improved
- `update_reference` docstring added

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
