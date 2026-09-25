# Architecture Overview

## High-Level Design

Reflective Lantern follows a layered architecture separating concerns into distinct packages:

```
┌─────────────────────────────────────────┐
│             API / CLI Layer             │
├─────────────────────────────────────────┤
│           Application Layer            │
│  (feature_store, model_registry, etc.) │
├─────────────────────────────────────────┤
│          Infrastructure Layer           │
│  (retry, circuit_breaker, token_bucket)│
├─────────────────────────────────────────┤
│           Observability Layer           │
│  (metrics_collector, alerting, profiler)│
└─────────────────────────────────────────┘
```

## Module Responsibilities

### Infrastructure
- **retry** — exponential backoff with jitter for transient failures
- **circuit_breaker** — protect downstream services from cascade failures
- **token_bucket** — rate-limit outbound requests per client key
- **task_queue** — priority-ordered background work execution

### Observability
- **metrics_collector** — counters, gauges, histograms for runtime state
- **alerting** — threshold-based rule evaluation with cooldown periods
- **profiler** — wall-clock decorators for latency tracking
- **health_check** — composable readiness/liveness probe registry
- **audit_log** — immutable append-only record of actor actions

### Data & ML
- **feature_store** — versioned feature sets for model serving
- **model_registry** — lifecycle management (staging → production → archived)
- **data_versioning** — SHA-256 checksummed snapshot lineage
- **data_augmentation** — text and numeric augmentation for training pipelines
- **experiment_tracker** — deterministic A/B variant assignment

### Utilities
- **pagination** — offset and cursor-based page helpers
- **event_bus** — synchronous pub/sub with wildcard subscriptions
- **correlation_id** — thread-local trace identifiers
- **geo_utils** — haversine distance, bounding boxes, nearest neighbour
- **compression** — zlib/gzip wrappers with JSON helpers
- **cost_estimator** — USD cost projection from resource specs
- **shadow_mode** — parallel shadow traffic comparison
- **config_validator** — schema-based config validation before startup
- **notification_dispatcher** — multi-channel alerting with severity routing
- **webhook_handler** — HMAC-verified inbound webhook processing

## Data Flow

```
Request → correlation_id → API handler
                               │
                    ┌──────────┴──────────┐
                    │                     │
               Feature lookup        Model inference
               (feature_store)       (model_registry)
                    │                     │
                    └──────────┬──────────┘
                               │
                         Response + audit_log entry
```

## Threading Model

All stateful components (MetricsRegistry, TokenBucket, TaskQueue) use
`threading.Lock` or `threading.Condition` for thread safety. The
`correlation_id` module uses `threading.local` for per-thread isolation.

## Memory Optimisation

Several hot-path classes use `__slots__` or `slots=True` on dataclasses to
reduce per-instance overhead:

- `CircuitBreaker` — six named slots prevent accidental attribute sprawl.
- `TokenBucket` — `@dataclass(slots=True)` cuts dict overhead for the four
  state fields that are updated on every consume/refill cycle.

## Resilience Patterns

### Retry with Exponential Backoff

`app/retry` wraps callables with configurable `max_attempts`, `base_delay`,
`backoff` multiplier, and `jitter`.  The delay formula is:

```
sleep = base_delay * backoff^(attempt - 1) + random(0, jitter)
```

`retry_on_network_error` is a convenience decorator pre-configured for
`OSError`/`TimeoutError` scenarios.

### Circuit Breaker

`app/circuit_breaker` moves through `CLOSED → OPEN → HALF_OPEN → CLOSED`.
The `is_closed` and `is_half_open` properties enable callers to branch on
state without importing `CircuitState`.

### Token Bucket Rate Limiting

`PerKeyTokenBucket` allocates an independent `TokenBucket` per key on first
use.  `fill_ratio` (0.0–1.0) exposes bucket fullness for dashboards without
requiring lock-held inspection of internal fields.
