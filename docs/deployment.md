# Deployment Guide

## Prerequisites

- Python 3.11+
- pip 23+
- Docker 24+ (optional, for containerised deployments)

## Local Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest -q
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `APP_ENV` | Yes | — | `dev` or `prod` |
| `APP_HOST` | No | `0.0.0.0` | Bind address |
| `APP_PORT` | No | `8000` | HTTP port |
| `LOG_LEVEL` | No | `INFO` | Python log level |
| `SENTRY_DSN` | No | — | Error tracking DSN |

## Docker

```bash
# Build
docker build -t reflective-lantern:latest .

# Run
docker run -e APP_ENV=prod -p 8000:8000 reflective-lantern:latest
```

### Multi-stage Dockerfile skeleton

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib /usr/local/lib
COPY app/ ./app/
CMD ["python", "-m", "app"]
```

## CI/CD

The repository ships a GitHub Actions workflow at `.github/workflows/ci.yml`
that runs on every push and pull request:

1. `pip install -e ".[dev]"`
2. `ruff check .`
3. `pytest -q --tb=short`

To deploy after CI passes, tag the commit:

```bash
git tag v1.2.3
git push origin v1.2.3
```

The release workflow (`.github/workflows/release.yml`) will build and
publish a Docker image tagged with the version.

## Health Probes

Once running, the following endpoints are expected:

| Path | Purpose |
|------|---------|
| `GET /health/live` | Liveness — returns 200 while the process is up |
| `GET /health/ready` | Readiness — runs the `HealthRegistry` and returns 503 on failure |
| `GET /metrics` | Prometheus-compatible metrics (if enabled) |

## Rolling Upgrades

1. Deploy new version alongside the old with a distinct port.
2. Run `/health/ready` on the new instance.
3. Shift traffic via load-balancer weight rules.
4. Drain and terminate old instance.
