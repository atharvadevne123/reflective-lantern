# Signal-Forge

![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/signal-forge-ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Real-time financial market regime detection and portfolio stress testing API.**

XGBoost/LightGBM/RandomForest ensemble with 5-factor feature engineering, FAISS-based historical regime lookup, and automated KS-test drift monitoring.

---

## Features

- **Market Regime Detection** — classify assets into bull / bear / sideways / volatile using a soft-voting ensemble (XGBoost + LightGBM + RandomForest)
- **Portfolio Risk Scoring** — 0–1 risk score combining volatility, momentum, beta, and regime
- **5-Factor Feature Pipeline** — rolling volatility (annualised), momentum, volume ratio, market correlation, beta
- **FAISS Pattern Search** — find the k most similar historical market periods for context
- **KS-Test Drift Monitoring** — two-sample KS test per feature; drift events logged to PostgreSQL
- **Airflow DAG** — weekly automated retraining pipeline: fetch → train → drift-check → FAISS rebuild
- **5-Fold CV + AUC-ROC** — model evaluation with stratified cross-validation
- **FastAPI** — typed Pydantic v2 request/response schemas, OpenAPI docs at `/docs`
- **Docker + PostgreSQL** — one-command deployment via `docker-compose up`

---

## Quick Start

```bash
# Clone and enter
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern/signal-forge

# Configure environment
cp .env.example .env

# Start services
docker-compose up --build

# Predict regime
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "close": 195.0, "volume": 80000000.0}'
```

---

## API Reference

### `POST /predict`

Classify market regime and return portfolio risk score.

**Request:**
```json
{
  "ticker": "AAPL",
  "close": 195.0,
  "volume": 80000000.0,
  "market_return": 0.005
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "regime": "bull",
  "confidence": 0.72,
  "risk_score": 0.31,
  "volatility": 0.18,
  "momentum": 0.04,
  "correlation": 0.65,
  "volume_ratio": 1.12,
  "beta": 1.05,
  "similar_periods": [],
  "model_version": "1.0.0"
}
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Regime classification + risk score |
| `/health` | GET | Service health status |
| `/metrics` | GET | Request count, uptime, drift summary |
| `/version` | GET | API version |
| `/drift-check` | POST | Trigger KS-test drift detection |

---

## Architecture

```
Client ──► FastAPI ──► Feature Pipeline (sklearn)
                            ├─ VolatilityFeature
                            ├─ MomentumFeature
                            ├─ VolumeRatioFeature
                            ├─ MarketCorrelationFeature
                            └─ BetaFeature
                       ──► Ensemble Model (XGB + LGB + RF)
                       ──► FAISS Pattern Search
                       ──► PostgreSQL (predictions + drift)
```

Full diagram: `screenshots/architecture.txt`

---

## Setup (Local)

```bash
cd signal-forge
pip install -r requirements.txt
cp .env.example .env  # edit DATABASE_URL for local SQLite
uvicorn app.main:app --reload
```

---

## Testing

```bash
pip install pytest pytest-asyncio httpx
python -m pytest tests/ -v --tb=short
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
