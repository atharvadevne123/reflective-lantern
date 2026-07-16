# Suite-Cast

![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Hotel booking demand forecasting and dynamic pricing API.** Suite-Cast predicts booking demand probability for a given room-night using an XGBoost + LightGBM ensemble, then converts that demand score into a dynamically priced room-rate suggestion. Every prediction is logged to a database, and a KS-test drift monitor compares live score distributions against the training reference so operators know when to retrain.

## Why

Hotels leave revenue on the table with static rate cards. Demand for a room-night varies with lead time, seasonality, day of week, occupancy trends, competitor pricing, and local events. Suite-Cast models these signals and returns a demand tier (low / medium / high) plus a suggested rate in a 0.7×–1.6× band around your base rate — the standard revenue-management envelope.

## Features

- **Ensemble model** — XGBoost + LightGBM averaged, each validated with 5-fold CV (AUC-ROC)
- **18-feature sklearn Pipeline** — lead-time buckets, seasonality index, competitor rate ratio, YoY occupancy delta, weekend×summer interaction, ordinal encodings
- **Dynamic pricing** — demand score drives a bounded price multiplier on the base rate
- **Model monitoring** — every prediction persisted via SQLAlchemy; `/metrics` runs a KS-test drift check against the training reference distribution
- **Automated retraining** — Airflow DAG with champion/challenger promotion gated on AUC improvement
- **Production API hygiene** — Pydantic validation, rate limiting, correlation-ID tracing, structured JSON logs, OpenAPI docs
- **Docker-first** — single `docker compose up` brings up the API and PostgreSQL

