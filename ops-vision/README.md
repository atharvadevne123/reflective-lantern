# Ops-Vision

[![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-192%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

SRE ML platform for real-time incident prediction, alert classification, and
performance anomaly detection. Ops-Vision scores live service telemetry with a
soft-voting ensemble, retrieves the matching remediation runbook, watches its own
input distribution for drift, and forecasts the next 24 hours of incident load.

---

## Why this exists

On-call engineers find out about degradation when the pager fires — after users
are already affected. Ops-Vision moves that signal earlier by treating incident
onset as a supervised classification problem over ordinary telemetry every
service already emits (CPU, memory, error rate, p99 latency, request rate, disk
I/O). Three things make the prediction usable rather than merely interesting:

- **It explains itself.** Every positive prediction carries a severity band and
  a retrieved runbook, so the responder gets a next action, not just a score.
- **It knows when it is wrong.** A KS-test drift monitor watches the live
  feature distribution against a reference window; when production drifts away
  from training, that is surfaced rather than silently degrading accuracy.
- **It retrains itself.** An Airflow DAG retrains nightly on recent production
  data and only promotes the candidate if it clears an AUC-ROC gate.

---

## Architecture

![Architecture](screenshots/architecture.svg)

| Layer | Component | Responsibility |
|---|---|---|
| Ingest | `app/schemas.py` | Pydantic validation of the six telemetry metrics |
| API | `app/api/v1/routes.py` | Eight versioned endpoints under `/api/v1` |
| Middleware | `app/middleware.py` | Correlation-ID propagation, per-IP rate limiting |
| Features | `app/features.py` | 4 engineered features + `RobustScaler`, as a sklearn `Pipeline` |
| Model | `app/model.py` | Soft-voting ensemble (XGBoost + LightGBM + RandomForest) |
| Retrieval | `app/faiss_index.py` | FAISS runbook index with brute-force cosine fallback |
| Monitoring | `app/monitoring.py` | KS-test drift detection over sliding windows |
| Forecasting | `app/forecasting.py` | Holt double exponential smoothing, 24h horizon |
| Persistence | `app/database.py`, `app/crud.py` | SQLAlchemy models, pooled PostgreSQL access |
| Retraining | `pipelines/retrain_dag.py` | Nightly Airflow DAG with AUC promotion gate |

### Feature engineering

The pipeline expands six raw metrics into ten features before scaling:

| Feature | Definition | Rationale |
|---|---|---|
| `resource_pressure` | `(0.6·cpu + 0.4·mem) / 100` | Single composite saturation signal |
| `latency_err_ratio` | `latency_p99 / max(error_rate, 0.001)` | Separates slow-but-healthy from failing |
| `throughput_pressure` | `rps · (1 + disk_io/100)` | Load weighted by I/O contention |
| `log_latency_p99` | `log1p(latency_p99)` | Compresses the heavy right tail |

The denominator clamp in `latency_err_ratio` is deliberate — error rate is
legitimately zero on healthy services, and an unguarded division would produce
`inf` and poison the scaler.

---

## Quick start

```bash
# 1. Clone and enter the project
cd ops-vision

# 2. Install dependencies
make install-dev

# 3. Copy the environment template
cp .env.example .env

# 4. Run the test suite
make test

# 5. Start the API
make run
```

The API is then live at <http://localhost:8000>, with interactive docs at
<http://localhost:8000/docs>.

### Docker

```bash
make docker-up     # API + PostgreSQL 16
make docker-down   # tear down
```

The API container waits on a Postgres healthcheck before starting, so a single
`docker compose up` is enough for a cold start.

---

## API reference

All business endpoints are versioned under `/api/v1`. Liveness probes
(`/health`, `/version`) are unversioned so load balancers never need updating.

### `POST /api/v1/predict`

Scores one telemetry observation.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "service_name": "payments-api",
    "cpu_usage_pct": 85.0,
    "memory_usage_pct": 88.0,
    "error_rate_per_min": 62.0,
    "latency_p99_ms": 1450.0,
    "request_rate_per_sec": 45.0,
    "disk_io_util_pct": 80.0
  }'
```

```json
{
  "service_name": "payments-api",
  "predicted_incident": true,
  "predicted_severity": "critical",
  "confidence": 0.94,
  "model_version": "1.0.0",
  "runbook_hint": "High CPU Mitigation",
  "timestamp": "2026-08-26T14:32:00Z"
}
```

Severity bands map from confidence: `≥0.90` critical, `≥0.75` high,
`≥0.50` medium, below that low.

### `POST /api/v1/predict/batch`

Scores up to 500 observations in one call, returning results in submission
order. Batching amortises the model and pipeline lookup across the request,
which matters when backfilling historical telemetry.

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"items": [{ "service_name": "payments-api", "cpu_usage_pct": 85.0,
        "memory_usage_pct": 88.0, "error_rate_per_min": 62.0,
        "latency_p99_ms": 1450.0, "request_rate_per_sec": 45.0,
        "disk_io_util_pct": 80.0 }]}'
```

### `GET /api/v1/incidents`

Lists persisted incidents most-recent-first, with `limit` (1–500), `offset`,
and an optional exact-match `service_name` filter.
