# Changelog

## [1.0.0] - 2026-10-14

### Added
- FastAPI service with `/api/v1/predict`, `/api/v1/portfolio/risk`, `/api/v1/drift`, `/api/v1/health`, `/api/v1/metrics`, `/api/v1/train` endpoints
- XGBoost + LightGBM + RandomForest soft-voting ensemble
- 22-feature engineering pipeline: RSI, MACD, Bollinger Bands, ATR, volume ratio, lag returns, rolling volatility, SMA ratios, momentum, spread, range, price position
- Portfolio risk analytics: VaR, Expected Shortfall, Sharpe ratio, max drawdown
- KS-test drift detection per feature column
- SQLAlchemy ORM with PredictionLog, DriftLog, ModelMetrics tables
- Champion/challenger Airflow retraining DAG
- Thread-safe TTL/LRU prediction cache
- Rate limiting middleware (200 req/min per IP)
- Correlation ID middleware for distributed tracing
- GZip response compression
- Pydantic v2 input validation
- 5-fold stratified cross-validation AUC-ROC
- Docker + docker-compose with PostgreSQL
- GitHub Actions CI (ruff lint + pytest, Python 3.11/3.12)
- Alembic migration support
