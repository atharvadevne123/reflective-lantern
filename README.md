# Logistics-Flow

![CI](https://github.com/atharvadevne123/Logistics-Flow/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Last-mile delivery time prediction and logistics optimization API using an
XGBoost + LightGBM + RandomForest ensemble, with route risk scoring, carrier
performance analytics, KS-test drift monitoring, and Airflow retraining.

![Architecture](screenshots/architecture.png)

---

## Overview

Logistics-Flow estimates how long a parcel will take to reach its destination,
given carrier, distance, weight, route type, and dispatch time. It is built as
a production service rather than a notebook: every prediction is validated,
logged, and monitored for distribution drift, and the model retrains itself on
a weekly Airflow schedule.

**What it does**

- Predicts delivery duration in minutes with an ensemble confidence score
- Scores carrier-specific delay risk (DHL, FedEx, UPS, USPS, Amazon)
- Engineers 13 features including cyclical time encoding and distance buckets
- Detects feature drift with a two-sample Kolmogorov–Smirnov test
- Logs every inference to SQLite (dev) or PostgreSQL (prod) for auditing
- Retrains automatically when 30 days of fresh data accumulate

---

## Setup

### Local

```bash
git clone https://github.com/atharvadevne123/Logistics-Flow
cd Logistics-Flow

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env

uvicorn app.main:app --reload
```

The API is then available at `http://localhost:8000`, with interactive
OpenAPI docs at `http://localhost:8000/docs`.

On first start the service trains a model on synthetic data and writes
`model.joblib`, `feature_pipeline.joblib`, and `metrics.json`.

### Docker

```bash
docker compose up --build
```

This starts the API on port 8000 alongside a PostgreSQL 15 instance, with the
model baked into the image at build time.

---
