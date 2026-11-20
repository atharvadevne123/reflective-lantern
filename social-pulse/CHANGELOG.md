# Changelog

All notable changes to Social-Pulse are documented here.

## [Unreleased] — 2026-11-20

### Added
- `PostIndex.bulk_add()` for batch vector insertion in `faiss_index.py`
- `PostIndex.clear()` for resetting the index state in `faiss_index.py`
- `is_valid_level()` utility to `logging_config.py` for validating log level strings
- `ContentSecurityPolicyMiddleware` to `middleware.py` with configurable CSP directives
- `SimilaritySearchRequest` Pydantic schema to `schemas.py`
- `is_valid_model_version()` semver validator to `config.py`
- `is_alpha_text()` and `word_count()` helpers to `validators.py`
- `avg_score()` and `get_trending_hashtags()` to `trend_scorer.py`
- `sanitize_text()` and `safe_json_dumps()` to `utils.py`
- `FeatureCache.get_all_keys()` for key introspection in `feature_store.py`
- `get_feature_distribution()` descriptive stats helper to `monitoring.py`
- `compute_confidence_distribution()` to `metrics.py`
- `get_model_by_id()` lookup to `model_registry.py`
- `sentiment_score()` rule-based sentiment signal to `features.py`
- `char_ngram_density()` lexical diversity metric to `features.py`
- `rate_per_minute()` convenience method to `RateLimitMiddleware`
- `RateLimitExceededError` and `InvalidFeatureVectorError` custom exceptions
- `table_names()` introspection helper to `database.py`
- 38 new test methods across 20 test files; 60 total improvement commits

## [1.0.0] — 2026-10-14

### Added
- FastAPI application with `/predict`, `/health`, `/metrics`, `/drift`, `/train`, `/version` endpoints
- LightGBM + XGBoost + RandomForest + LogisticRegression VotingClassifier ensemble
- NLP feature pipeline: 17 hand-crafted stats + TF-IDF bigrams (200 features)
- Cyclical hour-of-day and day-of-week encoding
- Follower count log-scale bucketing (7 buckets)
- SQLAlchemy models: `predictions`, `drift_records`, `training_runs`
- KS-test drift detection across 6 key features
- Airflow DAG for weekly drift-triggered retraining
- 5-fold stratified cross-validation with AUC-ROC scoring
- Correlation ID middleware for request tracing
- Pydantic input validation with field-level constraints
- Docker and docker-compose setup with PostgreSQL
- pytest test suite: 60+ tests across API, model, features, and monitoring
- GitHub Actions CI: lint + test
- Architecture diagram
