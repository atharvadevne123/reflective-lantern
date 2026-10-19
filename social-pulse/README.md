# Social-Pulse

![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)

Real-time viral social media content prediction API using an LightGBM + XGBoost + RandomForest
ensemble with NLP features, KS-test drift monitoring, and Airflow-scheduled retraining.

---

## Overview

Social-Pulse predicts whether a social media post will go viral (>1000 engagements within 24 h),
given the post text, author follower count, media presence, and posting time.

**What it does**

- Predicts viral probability (0–1) and binary label with ensemble confidence score
- Engineers 17+ NLP features: TF-IDF bigrams, cyclical time encoding, hashtag/emoji density, follower bucketing
- Logs every inference to SQLite (dev) or PostgreSQL (prod) for auditing
- Detects feature drift with Kolmogorov–Smirnov tests across 6 key features
- Retrains automatically on Sunday 02:00 UTC when drift is detected or 1 000 new samples accumulate

---

## Quick Start

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern/social-pulse

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`. Interactive docs at `/docs`.

### Docker

```bash
docker compose up --build
```

---

## API Reference

### POST /predict

Predict whether a post will go viral.

```json
{
  "text": "Just dropped our new product launch! #innovation #tech @everyone",
  "author_followers": 50000,
  "has_media": true,
  "has_link": true,
  "hashtag_count": 2,
  "mention_count": 1,
  "hour_of_day": 18,
  "day_of_week": 2
}
```

**Response**
```json
{
  "request_id": "a1b2c3d4-...",
  "viral_probability": 0.73,
  "viral_label": true,
  "ensemble_confidence": 0.46,
  "model_version": "1.0.0",
  "latency_ms": 4.2
}
```

### GET /health
Returns service health, model status, and uptime.

### GET /metrics
Returns aggregate stats: total predictions, viral rate, avg confidence, avg latency.

### GET /drift
Returns recent KS-test drift detection results.

### POST /drift/check
Triggers a drift check across reference and current windows.

### POST /train
Retrains the model (uses synthetic data by default; swap in real labelled data for production).

### GET /version
Returns API and model version strings.

---

## Architecture

```
Client → FastAPI → Feature Pipeline → Ensemble Model → PostgreSQL
                       │                    │
               TextStatTransformer    LightGBM + XGBoost
               TF-IDF (200 features)  + RandomForest
                                            │
                                      KS-test Drift Monitor
                                            │
                                      Airflow Retrain DAG
```

See `screenshots/architecture.txt` for a detailed ASCII diagram.

---

## Testing

```bash
pip install pytest pytest-asyncio httpx
pytest social-pulse/tests/ -v --tb=short
```

---

## Project Structure

```
social-pulse/
├── app/
│   ├── __init__.py        # Package metadata
│   ├── database.py        # SQLAlchemy models and session
│   ├── features.py        # NLP feature engineering pipeline
│   ├── model.py           # Ensemble training and inference
│   ├── monitoring.py      # KS-test drift detection and logging
│   └── main.py            # FastAPI application
├── pipelines/
│   └── retrain_dag.py     # Airflow DAG for weekly retraining
├── tests/
│   ├── conftest.py        # Shared fixtures
│   ├── test_api.py        # Endpoint tests
│   ├── test_model.py      # Model tests
│   ├── test_features.py   # Feature pipeline tests
│   └── test_monitoring.py # Drift detection tests
├── scripts/
│   └── generate_diagram.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT
