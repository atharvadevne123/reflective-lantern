# Reflective Lantern

A growing collection of production-quality Python utility modules for
ML-serving platforms, API services, and data pipelines.

## Modules

### Infrastructure
| Module | Description |
|--------|-------------|
| `app/retry.py` | Exponential-backoff retry decorator with jitter |
| `app/circuit_breaker.py` | CLOSED / OPEN / HALF_OPEN state machine |
| `app/token_bucket.py` | Thread-safe per-key rate limiter |
| `app/task_queue.py` | Priority task queue with worker threads |
| `app/webhook_handler.py` | HMAC-verified inbound webhook dispatcher |
| `app/config_validator.py` | Schema-based configuration validation |

### Observability
| Module | Description |
|--------|-------------|
| `app/metrics_collector.py` | Counters, gauges, and histograms |
| `app/alerting.py` | Threshold alert rules with cooldown |
| `app/profiler.py` | Wall-clock timing and call-stats decorators |
| `app/health_check.py` | Composable readiness/liveness probe registry |
| `app/audit_log.py` | Immutable append-only structured audit trail |
| `app/notification_dispatcher.py` | Multi-channel severity-routed notifications |
| `app/correlation_id.py` | Thread-local request trace identifiers |

### Data & ML
| Module | Description |
|--------|-------------|
| `app/feature_store.py` | Versioned feature set storage |
| `app/model_registry.py` | ML model lifecycle (staging / production / archived) |
| `app/data_versioning.py` | SHA-256 checksummed snapshot lineage |
| `app/data_augmentation.py` | Text and numeric training data augmentation |
| `app/experiment_tracker.py` | Deterministic A/B variant assignment |
| `app/batch_processor.py` | Chunked batch execution with callbacks |
| `app/shadow_mode.py` | Parallel shadow traffic comparison |
| `app/cost_estimator.py` | USD cost projection from resource specs |

### Energy Analytics
| Module | Description |
|--------|-------------|
| `app/tariff.py` | Flat, time-of-use and tiered electricity pricing |
| `app/load_profile.py` | Base load, load factor, ramp rate, profile class |
| `app/weather_normalization.py` | Degree-day adjustment separating weather from efficiency |
| `app/demand_response.py` | Baseline load, curtailment and event settlement |
| `app/power_quality.py` | Power factor, reactive power, voltage imbalance |
| `app/solar.py` | PV generation, self-consumption split, payback |
| `app/battery.py` | Storage dispatch, peak shaving, capacity sizing |

See [docs/energy_analytics.md](docs/energy_analytics.md) for the full reference.

### Utilities
| Module | Description |
|--------|-------------|
| `app/pagination.py` | Offset and cursor-based pagination helpers |
| `app/event_bus.py` | Synchronous pub/sub with wildcard subscriptions |
| `app/geo_utils.py` | Haversine distance, bounding boxes, nearest neighbour |
| `app/compression.py` | zlib / gzip compress/decompress with JSON helpers |

## Quick Start

```bash
pip install -e ".[dev]"
pytest -q
```

## Common Patterns

```python
# Retry with exponential backoff
from app.retry import retry


@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict: ...


# Circuit breaker
from app.circuit_breaker import CircuitBreaker

cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
result = cb.call(fetch_data, url)

# Per-request correlation IDs
from app.correlation_id import correlation_context

with correlation_context() as cid:
    process_request()  # all logs carry cid

# Rate limiting
from app.token_bucket import PerKeyTokenBucket

bucket = PerKeyTokenBucket(capacity=100, refill_rate=10)
if bucket.consume(client_id):
    handle_request()
```

## Developer Scripts

```bash
make benchmark   # micro-benchmark all modules
make seed        # seed dev data fixtures
make test-cov    # tests with coverage report
```

## Testing

```bash
make test          # full suite
make test-cov      # with coverage report
make check         # lint + typecheck + tests, as CI runs them
```

The root suite holds 5767 tests and is green. Sub-project suites live under
their own directories (`quake-net/tests/`, `veritas-rag/tests/`, and so on)
and are run from within each project:

```bash
cd quake-net && pytest tests/ -q
```

Some sub-project suites have pre-existing failures — see
[Known Issues](docs/known_issues.md) for the cause and suggested fix of each.

## Docs

- [Architecture](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Energy Analytics](docs/energy_analytics.md)
- [Deployment](docs/deployment.md)
- [Monitoring](docs/monitoring.md)
- [Known Issues](docs/known_issues.md)

## License

MIT
