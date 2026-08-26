# Ops-Vision

[![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-192%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

SRE ML platform for real-time incident prediction, alert classification, and
performance anomaly detection. Ops-Vision scores live service telemetry with a
soft-voting ensemble, retrieves the matching remediation runbook, watches its own
input distribution for drift, and forecasts the next 24 hours of incident load.

---

## Why this exists

On-call engineers find out about degradation when the pager fires — after users
are already affected. Ops-Vision moves that signal earlier by treating incident
onset as a supervised classification problem over ordinary telemetry every
service already emits (CPU, memory, error rate, p99 latency, request rate, disk
I/O). Three things make the prediction usable rather than merely interesting:

- **It explains itself.** Every positive prediction carries a severity band and
  a retrieved runbook, so the responder gets a next action, not just a score.
- **It knows when it is wrong.** A KS-test drift monitor watches the live
  feature distribution against a reference window; when production drifts away
  from training, that is surfaced rather than silently degrading accuracy.
- **It retrains itself.** An Airflow DAG retrains nightly on recent production
  data and only promotes the candidate if it clears an AUC-ROC gate.

---

## Architecture

![Architecture](screenshots/architecture.svg)

| Layer | Component | Responsibility |
|---|---|---|
| Ingest | `app/schemas.py` | Pydantic validation of the six telemetry metrics |
| API | `app/api/v1/routes.py` | Eight versioned endpoints under `/api/v1` |
| Middleware | `app/middleware.py` | Correlation-ID propagation, per-IP rate limiting |
| Features | `app/features.py` | 4 engineered features + `RobustScaler`, as a sklearn `Pipeline` |
| Model | `app/model.py` | Soft-voting ensemble (XGBoost + LightGBM + RandomForest) |
| Retrieval | `app/faiss_index.py` | FAISS runbook index with brute-force cosine fallback |
| Monitoring | `app/monitoring.py` | KS-test drift detection over sliding windows |
| Forecasting | `app/forecasting.py` | Holt double exponential smoothing, 24h horizon |
| Persistence | `app/database.py`, `app/crud.py` | SQLAlchemy models, pooled PostgreSQL access |
| Retraining | `pipelines/retrain_dag.py` | Nightly Airflow DAG with AUC promotion gate |
