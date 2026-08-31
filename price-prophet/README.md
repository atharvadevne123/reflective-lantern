# Price-Prophet

[![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/reflective-lantern/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dynamic pricing and demand forecasting service using an XGBoost/scikit-learn ensemble pipeline.

## Overview

Price-Prophet is a FastAPI service that predicts optimal product prices and forecasts demand
using historical sales data, price elasticity modelling, and market context features.

### Features

- XGBoost + Linear ensemble for demand forecasting
- Price elasticity estimation with confidence intervals
- Backtesting framework for strategy evaluation
- REST API with batch prediction support
- SQLite/PostgreSQL storage with Alembic migrations
- KS-test distribution drift detection
- Full CLI for training and inference

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Start the API server
uvicorn app.api.main:app --reload --port 8000
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health and model status |
| `/predict` | POST | Single price/demand prediction |
| `/batch-predict` | POST | Batch prediction |
| `/metrics` | GET | Model performance metrics |

## Architecture

```
price-prophet/
├── app/
│   ├── api/          # FastAPI routes and schemas
│   ├── models/       # ML model implementations
│   ├── pricing/      # Pricing strategy logic
│   ├── evaluation/   # Backtesting and metrics
│   └── data/         # Data loading and preprocessing
├── tests/            # Pytest test suite
└── scripts/          # Training and export scripts
```

## Testing

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
