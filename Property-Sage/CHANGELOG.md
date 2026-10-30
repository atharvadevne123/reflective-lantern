# Changelog

All notable changes to Property-Sage are documented here.

## [1.2.0] – 2026-10-30

### Added
- `get_recent_runs`, `log_model_version`, `list_experiments` in `experiment_tracking`
- `calculate_confidence_score`, `batch_predict`, `get_model_version` in `model`
- `validate_year_built_value`, `sanitize_request_id`, `normalize_neighborhood` in `validators`
- `format_currency`, `clamp`, `slugify`, `percentage_change`, `is_finite_positive` in `utils`
- `compute_cap_rate`, `property_age_category`, `compute_price_per_sqft`, `compute_gross_yield` in `features`
- `encode_property_features`, `sqft_percentile`, `property_type_to_multiplier` in `features`
- `neighborhood_price_stats`, `neighborhood_benchmark` in `stats`
- `format_prediction_response`, `format_metrics_response` in `formatters`
- `get_prediction_count`, `get_recent_predictions` in `database`
- `yield_category`, `anomaly_score`, `detect_outlier_predictions`, `prediction_summary_stats` in `monitoring`
- `get_drift_history` in `monitoring`
- `property_valuation_report` in `reports`
- `is_healthy`, `get_unhealthy_components` in `health`
- `NeighborhoodPriceIndex` class in `price_index`
- `PredictionCache` LRU class in `cache`
- Comprehensive test coverage for all new utilities (30+ new test modules)

## [1.0.0] – 2026-09-10

### Added
- XGBoost + LightGBM + RandomForest VotingRegressor ensemble (weights 0.4/0.4/0.2)
- sklearn `Pipeline` with 11 engineered features (property age, age², sqft/bed, bath/bed, lot density, neighbourhood and type encodings)
- FastAPI REST API under `/api/v1/` with correlation-ID and rate-limiting middleware
- SQLAlchemy ORM (`Prediction`, `ModelMetrics`, `DriftLog`) for SQLite dev / PostgreSQL prod
- Alembic migration scaffolding with initial schema revision
- KS-test drift detection on 24-hour rolling prediction window; auto-retrains when p < 0.05
- TF-IDF + cosine-similarity RAG pipeline over 10 neighbourhood market reports
- 5-fold cross-validation reporting R², RMSE, MAE for every training run
- File-based experiment tracking (JSONL run log) with `log_run`, `get_best_run`, `list_runs`
- Automated retraining endpoint (`POST /api/v1/retrain`) with API-key guard
- Structured JSON logging via `JsonFormatter`
- Docker Compose with health-checks for API + PostgreSQL services
- GitHub Actions CI (ruff lint + pytest) on push / PR
- 98 pytest tests covering features, model, monitoring, drift, RAG, retrain, health, and API endpoints
- Inference latency benchmarking script (`scripts/benchmark.py`)
- `.env.example` with all required environment variables
