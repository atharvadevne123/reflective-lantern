# Support-Sage

![CI](https://github.com/atharvadevne123/Support-Sage/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**IT helpdesk ticket triage API** — predicts priority (critical/high/medium/low), estimates resolution time, and retrieves similar historical tickets using FAISS-based RAG.

## Overview

Support-Sage ingests a raw helpdesk ticket (subject + description) and returns:

| Field | Type | Description |
|-------|------|-------------|
| `priority` | str | critical / high / medium / low |
| `confidence` | float | Model confidence (0–1) |
| `resolution_hours_estimate` | float | Predicted hours to resolution |
| `priority_probabilities` | dict | Per-class probabilities |

Built with an **XGBoost + LightGBM + RandomForest** soft-voting ensemble trained on TF-IDF (5000 n-grams) + 24 meta-features (urgency keywords, structural signals, category indicators).

## Architecture

![Architecture](screenshots/architecture.png)

## Quick Start

```bash
git clone https://github.com/atharvadevne123/Support-Sage
cd Support-Sage
pip install -r requirements.txt
cp .env.example .env
make train    # train model on synthetic data
make run      # start API on :8000
```

## Docker

```bash
docker-compose up --build
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## API Reference

### POST /api/v1/predict

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Cannot connect to VPN from home",
    "description": "Since this morning I am unable to establish VPN. Error code 800. Already restarted twice."
  }'
```

**Response:**
```json
{
  "ticket_id": "...",
  "priority": "high",
  "confidence": 0.78,
  "priority_probabilities": {"critical": 0.05, "high": 0.78, "medium": 0.14, "low": 0.03},
  "resolution_hours_estimate": 6.5,
  "model_version": "1.0.0"
}
```

### GET /api/v1/health

Returns `{"status": "ok", "model_loaded": true, "version": "1.0.0"}`.

### GET /api/v1/metrics

Returns model training metrics (AUC, feature count) and KS-test drift statistics across a 7-day window.

### POST /api/v1/train

Triggers retraining on fresh synthetic data and reloads the in-memory model cache.

## Tech Stack

| Component | Technology |
|-----------|----------|
| API | FastAPI + Uvicorn |
| ML | XGBoost + LightGBM + RandomForest (ensemble) |
| Features | sklearn TF-IDF + TruncatedSVD + meta-features |
| RAG | FAISS + TF-IDF/SVD embeddings |
| Drift | KS-test (scipy) |
| ORM | SQLAlchemy (SQLite dev / PostgreSQL prod) |
| Orchestration | Airflow DAG (weekly retrain) |
| CI | GitHub Actions (ruff + pytest) |
| Containers | Docker + docker-compose |

## Tests

```bash
make test
# 30+ tests across API, model, features, monitoring
```

## Model Details

- **Ensemble:** XGBoost (200 trees) + LightGBM (200 trees) + RandomForest (100 trees), soft voting
- **Features:** TF-IDF 5000 unigrams+bigrams + 24 meta features (urgency keywords, lengths, ratios)
- **CV:** 5-fold stratified cross-validation, AUC-ROC reported
- **Retraining:** Airflow DAG triggers weekly; also available via `/api/v1/train`
- **Drift:** KS-test on confidence scores, resolution-hour predictions, and word counts; `p < 0.05` flags drift

## Deployment

### Manual GitHub Push

```bash
GH_PAT=<your-token> bash deploy.sh
```

### Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

## License

MIT
