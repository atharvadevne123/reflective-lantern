# Monitoring & Observability

## Metrics

All runtime metrics are collected via `app.metrics_collector` and exposed
on the `/metrics` HTTP endpoint in Prometheus text format.

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests by method and status |
| `http_request_duration_seconds` | Histogram | Latency distribution |
| `active_workers` | Gauge | Current TaskQueue worker count |
| `circuit_breaker_state` | Gauge | 0=CLOSED, 1=OPEN, 2=HALF_OPEN |
| `token_bucket_rejections_total` | Counter | Rate-limit rejections per client key |
| `model_inference_duration_seconds` | Histogram | Model serving latency |

## Health Checks

Register checks via `app.health_check`:

```python
from app.health_check import CheckResult, check

@check("postgres")
def check_postgres():
    # run a SELECT 1
    return CheckResult(name="postgres", healthy=True)
```

The `/health/ready` endpoint runs all registered checks and returns:
- `200 OK` with `{"healthy": true}` when all pass
- `503 Service Unavailable` with a list of failures otherwise

## Alerting

Threshold-based alerts are managed by `app.alerting.AlertManager`.
Add rules at startup:

```python
from app.alerting import AlertManager, AlertRule, Severity

manager = AlertManager()
manager.add_rule(AlertRule(
    name="high_error_rate",
    metric="error_rate",
    threshold=0.05,
    operator=">",
    severity=Severity.CRITICAL,
    cooldown=300,
))
manager.add_handler(lambda alert: notify_oncall(alert))
```

## Distributed Tracing

Every request should propagate a correlation ID via the `X-Correlation-ID`
HTTP header. The `app.correlation_id` module stores it in thread-local
memory so all log lines within the request carry the same ID.

```python
from app.correlation_id import correlation_context

with correlation_context(request.headers.get("X-Correlation-ID")):
    handle_request()
```

## Log Format

All modules use the standard `logging` module at the `DEBUG`/`INFO`/`ERROR`
levels. Structured JSON logging is recommended in production:

```python
import logging, json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "level": record.levelname,
            "msg": record.getMessage(),
            "module": record.module,
        })
```

## Dashboards

Import the provided Grafana dashboard JSON from `docs/grafana/` to get
pre-built panels for:
- Request rate and latency (P50/P95/P99)
- Circuit breaker state transitions
- Token bucket rejection rate
- Worker queue depth
- Health check pass rate

## Runbook: Circuit Breaker Open

1. Check `circuit_breaker_state` metric — confirm it is `1` (OPEN).
2. Identify the downstream dependency that is failing.
3. Resolve the underlying issue (restart the dependency, fix network route).
4. The circuit transitions to HALF_OPEN automatically after `recovery_timeout`.
5. If the probe call succeeds, the circuit closes; otherwise it stays OPEN.
6. Do **not** manually reset the breaker — let the recovery timeout handle it.

## Runbook: High Memory Usage

1. Check `active_workers` — a large queue depth may indicate a processing backlog.
2. Profile with `app.profiler.get_stats()` to identify hot paths.
3. Increase `workers` in `TaskQueue` if CPU is available.
4. Consider enabling `compression` for large in-memory data structures.
