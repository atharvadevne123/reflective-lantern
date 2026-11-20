# News-Guard

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Real-time fake news and misinformation classification API using LightGBM-XGBoost-RandomForest ensemble with TF-IDF feature engineering, FAISS article similarity retrieval, and KS-drift detection.

## Overview

News-Guard classifies articles as **FAKE** or **REAL** using:

- **TF-IDF n-gram features** extracted from article headlines (300 features, unigrams + bigrams)
- **13 hand-crafted linguistic signals**: clickbait score, ALL-CAPS ratio, sensational/hedge words, source domain credibility, exclamation count, vocabulary richness
- **Source credibility** flags for 10 known reliable and 8 known unreliable domains

Predictions are logged to PostgreSQL. KS-drift detection compares live prediction distributions against a reference. A weekly Airflow DAG retrains automatically.

## Quick Start

```bash
cp .env.example .env
docker-compose up --build -d
```

## API Reference

### POST `/api/v1/predict`

```json
{
  "headline": "SHOCKING: Experts reveal secret government plot!!!",
  "body": "Sources claim explosive revelations...",
  "source_domain": "infowars.com"
}
```

Response:
```json
{
  "article_id": "a1b2c3d4e5f6",
  "prediction": "FAKE",
  "confidence": 0.923,
  "credibility_score": 0.077,
  "model_version": "1.0.0"
}
```

### GET `/api/v1/health` — liveness check  
### GET `/api/v1/metrics` — prediction stats + drift status  
### GET `/api/v1/drift` — on-demand KS-test  

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| ML | XGBoost + RandomForest (VotingClassifier) |
| Features | scikit-learn Pipeline + TF-IDF |
| Monitoring | KS-test drift detection |
| Database | SQLAlchemy + PostgreSQL / SQLite |
| Retraining | Airflow DAG |
| Container | Docker + docker-compose |
| Tests | pytest (5 files, 37 cases) |
| CI | GitHub Actions (ruff + pytest) |
