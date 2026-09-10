# Deploying Cyber-Guard

## Local (SQLite)

```bash
cp .env.example .env
make install
make migrate
make run
```

The service creates its tables and trains an initial model on first boot, so
`/api/v1/health` returns `healthy` without any further setup.

## Docker Compose (PostgreSQL)

```bash
docker-compose up --build
```

The API waits on the `db` service's health check before starting, so the
first request never races Postgres coming up.

## Production checklist

- **Set `DATABASE_URL` to PostgreSQL.** SQLite is fine for development but
  serialises writes, and every prediction writes a row.
- **Run migrations, do not rely on `create_tables()`.** The app calls
  `create_tables()` on boot for convenience; in production run
  `alembic upgrade head` as a release step so schema changes are versioned.
- **Set `RATE_LIMIT_PER_MINUTE` deliberately.** The limiter is in-process, so
  with N replicas the effective global limit is N times this value. For a
  true global limit, back `app/rate_limit.py` with Redis — the module is
  written so that substitution stays local to it.
- **Point `MODEL_PATH` at a shared volume, or set `S3_BUCKET`.** Otherwise
  each replica trains its own model on first boot and they will disagree.
- **Configure `REFERENCE_WINDOW_DAYS` to exceed your traffic seasonality.**
  A window shorter than a weekly cycle will report drift every weekend.

## Readiness and liveness

Point both probes at `GET /api/v1/health`. It checks the database as well as
the models and reports `degraded` when a dependency is down, so an instance
that has lost its database is taken out of rotation rather than continuing to
serve. The endpoint is exempt from rate limiting.

```yaml
readinessProbe:
  httpGet: { path: /api/v1/health, port: 8000 }
  initialDelaySeconds: 20   # first boot trains a model
  periodSeconds: 10
```

Set `initialDelaySeconds` generously: the first boot trains both the
classifier and the anomaly detector before the app reports ready.

## Monitoring

- `GET /api/v1/metrics?hours=24` — prediction volume, class mix, mean
  confidence.
- `GET /api/v1/drift` — KS test on the `src_bytes` distribution. Alert on
  `drift_detected: true` sustained across several checks; a single positive on
  a low-traffic window is usually noise.
- A rising `is_anomaly` rate on `/api/v1/predict` responses is the earliest
  signal of a novel attack pattern, since it needs no labels.
