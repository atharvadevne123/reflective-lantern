# Watt-Guard — Development Guide

## Prerequisites

- Python 3.11+
- `pip` or `uv`
- (Optional) Docker + Docker Compose for containerised runs
- (Optional) Apache Airflow 2.x for the retraining pipeline

## Local Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in config
cp .env.example .env

# 4. Initialise the dev database
make migrate

# 5. Start the API server with hot reload
make run
# → http://localhost:8000/docs
```

## Project Structure

```
app/
├── main.py           ← FastAPI application and route handlers
├── model.py          ← VotingRegressor ensemble, train/predict/load
├── features.py       ← 7-stage sklearn feature pipeline
├── anomaly.py        ← Z-score + IQR severity classifier
├── monitoring.py     ← KS-drift detection, prediction logging
├── database.py       ← SQLAlchemy models and session factory
├── schemas.py        ← Pydantic v2 request/response schemas
├── config.py         ← Pydantic Settings from environment
├── cache.py          ← TTL in-memory prediction cache
├── middleware.py     ← Rate limiting + correlation ID
├── regions.py        ← Grid region registry with lru_cache
├── similarity.py     ← Brute-force cosine similarity index
├── faiss_index.py    ← FAISS load-pattern index (with fallback)
├── validation.py     ← Input sanity checks (temporal, weather, load)
├── reporting.py      ← Cost/savings report generation
├── time_series.py    ← Moving averages and seasonal baseline helpers
├── logging_config.py ← JSON structured logging + correlation ID
├── aws_stub.py       ← boto3 S3 stub (local mirror fallback)
└── mlflow_stub.py    ← MLflow stub (JSONL local log fallback)

tests/               ← pytest suite (mirrors app/ structure)
pipelines/           ← Airflow DAG for weekly retraining
scripts/             ← CLI utilities (client, diagram generator)
docs/                ← Markdown documentation
```

## Running Tests

```bash
make test            # full suite
make coverage        # with HTML report at htmlcov/index.html
pytest tests/test_api.py -k predict -v   # single file / keyword filter
```

## Linting & Type Checking

```bash
make format          # auto-fix with ruff
make lint            # check only (CI mode)
make typecheck       # mypy strict pass on app/
make check           # lint + typecheck (local CI equivalent)
```

## Feature Pipeline

The sklearn `Pipeline` in `app/features.py` applies these steps in order:

| Step | Class | Output Features |
|---|---|---|
| 1 | `TemporalFeatureExtractor` | sin/cos hour, day, month; is_weekend, is_business_hour |
| 2 | `LagFeatureExtractor` | lag_1h … lag_168h |
| 3 | `RollingStatsExtractor` | mean/std/min/max over 3h, 6h, 24h |
| 4 | `WeatherFeatureExtractor` | heat_index, cooling/heating degree-hours, temp×humidity |
| 5 | `OccupancyFeatureExtractor` | occ×hvac proxy, log-occupancy |
| 6 | `DropNonNumeric` | drops object/string columns |
| 7 | `StandardScaler` | zero-mean unit-variance |

## Adding a New Endpoint

1. Add request/response schemas to `app/schemas.py`
2. Add the route to `app/main.py`
3. Add ≥1 happy-path and ≥1 error test in `tests/test_api.py`
4. Document it in `README.md` and `docs/API.md`

## Docker

```bash
make docker-up       # docker-compose up --build -d
make docker-down     # docker-compose down -v
```

The compose file starts an API container and a PostgreSQL 15 container. The API waits for the DB via `depends_on: condition: service_healthy`.

## Airflow

```bash
export AIRFLOW_HOME=~/airflow
airflow db init
airflow dags list   # verify watt_guard_weekly_retrain appears
airflow dags trigger watt_guard_weekly_retrain
```

## Environment Variables

See `.env.example` for the full reference. Key variables for local dev:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./watt_guard.db` | DB connection string |
| `MODEL_PATH` | `model.joblib` | Path to trained ensemble |
| `ANOMALY_MODEL_PATH` | `anomaly_model.joblib` | Path to IsolationForest |
| `API_PORT` | `8000` | FastAPI server port |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LOG_JSON` | `false` | Emit structured JSON logs |
| `RATE_LIMIT_PER_MINUTE` | `200` | Per-IP rate limit |
