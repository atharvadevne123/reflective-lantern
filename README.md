![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)
![Python Package](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/python-publish.yml/badge.svg)
![npm](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/npm-publish.yml/badge.svg)
![Bump Version](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/bump-version.yml/badge.svg)

> Smart building and industrial energy consumption forecasting and anomaly detection API.

[![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/reflective-lantern/actions)
[![Coverage](https://img.shields.io/badge/tests-59%20passed-brightgreen)](tests/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)

Every weekday at 9 AM CST, Reflective Lantern wakes up, picks one of @atharvadevne123’s
GitHub repositories, implements 60 improvements, runs tests, updates docs, pushes to main,
and sends an email digest — all with zero human intervention.

## Quick Start

```bash
git clone https://github.com/atharvadevne123/reflective-lantern.git
cd reflective-lantern
bash scripts/setup.sh          # install deps + pre-commit hooks
cp .env.example .env           # fill in your API keys
make test                      # verify everything works
```

**Required environment variables** (see `.env.example` for full list):

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for AI-powered features |
| `GH_PAT` | GitHub Personal Access Token (`repo` + `workflow` scopes) |
| `NOTION_API_KEY` | For Notion portfolio updates |
| `GMAIL_USER` + `GMAIL_APP_PASS` | For emailed run reports |

## What It Does

Each daily run:
1. **PRE-FLIGHT** — Fix failing CI workflows, merge open branches, create missing releases
2. **MODE SELECT** — IMPROVEMENT (most days) or INNOVATION (Wednesday wks 2 & 4)
3. **SELECT REPO** — deterministic daily rotation through the active portfolio
4. **ANALYSE** — read every source file, identify 60 improvements across 5 tiers
5. **IMPLEMENT** — one commit per change (security → tests → quality → DX → perf)
6. **VERIFY** — run full test suite; fix failures (2 attempts)
7. **PUSH** — directly to `main`
8. **NOTIFY** — PDF report emailed to devneatharva@gmail.com

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a full ASCII diagram.

```
reflective-lantern/
├── .claude/settings.json     ← CCR tool permissions
├── config/                    ← Settings, constants, logging
├── scripts/                   ← Standalone utility scripts
├── tests/                     ← pytest suite
├── docs/                      ← Architecture & operations docs
├── history/                   ← Per-repo JSON run logs
├── prompts/system_prompt.md  ← Cached agent instructions (3000+ tokens)
└── covers/                    ← SVG cover images for Notion
```

## Improvement Tiers

| Priority | Tier | Examples |
|----------|------|----------|
| 1 | Security / Correctness | Secrets → env vars, bare `except` → typed, SQL injection |
| 2 | Tests | `conftest.py`, happy path + 3 edge cases per endpoint |
| 3 | Code Quality | Type hints, docstrings, logging, refactor > 40-line functions |
| 4 | Developer Experience | CI/CD, Dockerfile, `.env.example`, `pyproject.toml`, README |
| 5 | Performance | `lru_cache`, N+1 fix, DB indexes, connection pooling |

## Utility Scripts

```bash
# Clone
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern && git checkout innovation/watt-guard

# Install
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker-compose up --build
```

## API Reference

### POST `/api/v1/predict`
Forecast energy consumption for a building.

```json
{
  "building_id": "bldg-001",
  "timestamp": "2025-06-01T14:00:00",
  "hour": 14,
  "day_of_week": 1,
  "month": 6,
  "temperature_c": 28.5,
  "humidity_pct": 60.0,
  "occupancy": 50,
  "hvac_state": 1,
  "consumption_kwh": 12.0
}
```

### POST `/api/v1/anomaly`
Detect anomalous consumption readings.

### POST `/api/v1/drift`
KS-test between reference and current distributions.

### GET `/api/v1/metrics`
Prediction counts, anomaly counts, drift events, model R2/MAE.

### GET `/api/v1/health`
API liveness and model load status.

### POST `/api/v1/train`
Train both models on synthetic seed data (for demo/CI).

### GET `/api/v1/version`
Return the running API version and build metadata.

```json
{"version": "1.1.0", "api": "v1", "model": "xgb+lgbm+rf"}
```

### GET `/api/v1/regions`
List all supported grid regions with peak load and timezone data.

```json
[
  {"id": "northeast", "name": "Northeast Grid", "peak_load_mw": 32000, "timezone": "America/New_York"},
  ...
]
```

## Feature Engineering

The 7-stage sklearn Pipeline computes:
- **Temporal**: hour/day/month cyclic sin-cos, is_weekend, is_business_hour
- **Lag**: 1h, 2h, 3h, 6h, 12h, 24h, 168h consumption lags
- **Rolling**: mean/std/min/max over 3h, 6h, 24h windows
- **Weather**: heat index, cooling/heating degree-hours, temp-humidity ratio
- **Occupancy**: occ×HVAC load proxy, log-occupancy density
- **DropNonNumeric**: removes non-numeric columns before scaling
- **StandardScaler**: zero-mean unit-variance normalisation

## Model Monitoring

Every prediction is logged to `prediction_logs`. Drift is checked via KS-test between a reference window (last 500 readings after training) and the current window. Anomaly scores from IsolationForest are logged to `anomaly_logs`.

## Retraining

The Airflow DAG `watt_guard_weekly_retrain` runs every Sunday. It fetches recent data, validates row count ≥ 500, retrains the model, gates on R2 ≥ 0.70, and deploys the new artefact.

## Testing

```bash
pytest tests/ -v
```

## License

MIT

## API Versioning

- **Scheduler**: Claude Code Cloud Routine (`cron 0 14 * * 1-5` = 9 AM CDT)
- **AI**: Claude Sonnet 4.6 with prompt caching
- **Repo ops**: GitHub REST API + git
- **Notifications**: Gmail SMTP
- **Portfolio**: Notion API + Anthropic SDK
