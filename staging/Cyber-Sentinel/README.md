# Cyber-Sentinel

[![CI](https://github.com/atharvadevne123/Cyber-Sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/Cyber-Sentinel/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time network intrusion and cyber attack detection API using an XGBoost-LightGBM-RandomForest ensemble with KS-drift monitoring, FAISS pattern matching, and Airflow retraining pipelines.

## Features

- **Ensemble ML**: Soft-voting classifier combining XGBoost, LightGBM, and Random Forest
- **Drift Detection**: KS-test monitoring on 25 network traffic features
- **FAISS Matching**: Sub-millisecond lookup of similar known attack signatures
- **5 Attack Classes**: normal, DoS, Probe, R2L, U2R
- **FastAPI**: Auto-generated OpenAPI docs at `/docs`
- **Airflow DAG**: Weekly automated retraining with drift-triggered alerting
- **PostgreSQL**: Persistent storage for events, predictions, and drift logs
- **Docker**: One-command deployment with `docker compose up`

## Quick Start

```bash
# 1. Clone
git clone https://github.com/atharvadevne123/Cyber-Sentinel.git
cd Cyber-Sentinel

# 2. Install
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env

# 4. Run locally
uvicorn app.main:app --reload

# 5. Or run with Docker
docker compose up -d
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health status and version |
| GET | `/version` | API version |
| POST | `/train` | Train ensemble on synthetic data |
| POST | `/predict` | Classify a network event |
| GET | `/metrics` | Prediction health + drift summary |
| GET | `/drift` | Run drift check and return alerts |
| GET | `/feature-importance` | RF feature importance scores |

### Predict Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "protocol": "tcp",
    "payload_bytes": 512,
    "duration_ms": 120.0,
    "src_bytes": 1024,
    "flags": "SA",
    "count": 10,
    "serror_rate": 0.1
  }'
```

Response:
```json
{
  "label": "attack",
  "confidence": 0.873,
  "attack_type": "dos",
  "class_probabilities": {"normal": 0.127, "dos": 0.873},
  "similar_patterns": [{"rank": 1, "similarity": 0.94, "attack_type": "dos"}]
}
```

## Architecture

```
Client ──► FastAPI (main.py)
               │
        ┌──────┼──────────────┐
        │      │              │
   features  model       monitoring
    (.py)   (.py)          (.py)
              │
   VotingClassifier (XGBoost | LightGBM | RF)
              │
   PostgreSQL via SQLAlchemy ORM
```

## Testing

```bash
make test
# or
pytest tests/ -v --cov=app
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).
