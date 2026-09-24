# Deployment

## Docker Compose (recommended for a single host)

```bash
cp .env.example .env
docker compose up --build -d
docker compose logs -f api
```

This starts the API on `:8000` and PostgreSQL 15 on `:5432`. The model is
trained during the image build, so the container serves traffic immediately
on start rather than training on first request.

## Database migrations

The schema is managed with Alembic.

```bash
export DATABASE_URL=postgresql://logistics:logistics@localhost:5432/logistics_flow
alembic upgrade head
```

`app.database.init_db()` also creates tables via `create_all` for local
development, but Alembic is the source of truth in production.

## Configuration

All settings resolve from the environment through `app/config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./logistics_flow.db` | Connection string |
| `MODEL_PATH` | `model.joblib` | Trained ensemble artifact |
| `FEATURE_PIPELINE_PATH` | `feature_pipeline.joblib` | Fitted feature pipeline |
| `METRICS_PATH` | `metrics.json` | CV metrics surfaced by `/metrics` |
| `LOG_LEVEL` | `INFO` | Root log level |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per-client request budget |
| `DRIFT_WINDOW` | `100` | Predictions compared in a drift check |

## Production checklist

- [ ] Restrict CORS: replace `allow_origins=["*"]` in `app/main.py`
- [ ] Move rate limiting to Redis or an API gateway if running >1 replica
- [ ] Point `DATABASE_URL` at managed PostgreSQL, not SQLite
- [ ] Run `alembic upgrade head` as a release step
- [ ] Wire `/api/v1/health` to readiness and liveness probes
- [ ] Schedule the Airflow DAG (`logistics_flow_retrain`) weekly
- [ ] Alert on `drift_logs` rows appearing

## Scaling notes

The service is stateless apart from the in-memory drift reference buffer,
which is rebuilt at startup. Horizontal scaling is safe; each replica keeps
its own buffer, so drift checks are per-replica samples of the same traffic.
