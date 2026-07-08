# Traffic-Pulse

[![CI](https://github.com/atharvadevne123/Traffic-Pulse/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/Traffic-Pulse/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688)
![License](https://img.shields.io/badge/license-MIT-green)

Real-time urban traffic congestion prediction and incident detection API using an
**XGBoost + LightGBM ensemble** with temporal feature engineering, route scoring,
and automated KS-test drift detection.

Traffic-Pulse classifies road segments into four congestion levels — `free`,
`moderate`, `congested`, `severe` — from live traffic telemetry (vehicle counts,
average speeds, incidents, weather), logs every prediction for monitoring, and
retrains automatically when feature drift is detected.

## Features

- **Ensemble ML** — XGBoost + LightGBM pipelines averaged at the probability level,
  trained with 5-fold stratified cross-validation (weighted one-vs-rest AUC-ROC).
- **26 engineered features** — cyclical hour/day-of-week encodings, peak-hour flags,
  lag features (1h/2h/4h), rolling means/stds (6h/24h), speed-volume ratios,
  incident density, and road-type encoding.
- **Drift detection** — two-sample Kolmogorov-Smirnov test per feature with
  drift events persisted to the database and surfaced in `/api/v1/metrics`.
- **Prediction logging** — every request/result stored via SQLAlchemy
  (SQLite in dev, PostgreSQL in prod).
- **Automated retraining** — drift-gated pipeline with AUC validation and
  model promotion (`pipelines/retrain_dag.py`, Airflow-compatible).
- **Production middleware** — correlation-ID propagation and response-time headers.

