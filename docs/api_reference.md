# API Reference

This document covers the public interface of every `app/` module.
All symbols listed below are exported via each module's `__all__`.

---

## app.retry

```python
retry(
    max_attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: tuple = (Exception,),
) -> Callable
```
Decorator. Retries the wrapped function up to `max_attempts` times on
any exception in `exceptions`, with exponential backoff.

---

## app.circuit_breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_timeout: float)
    def call(self, fn: Callable, *args, **kwargs) -> Any
    @property
    def state(self) -> CircuitState
```

`CircuitState` values: `CLOSED`, `OPEN`, `HALF_OPEN`.

---

## app.pagination

```python
class Page(Generic[T]):
    items: List[T]
    info: PageInfo

class PageInfo:
    page: int; per_page: int; total: int
    has_next: bool; has_prev: bool

def paginate(items, page, per_page) -> Page
def cursor_paginate(items, cursor, per_page) -> CursorPage
```

---

## app.event_bus

```python
class EventBus:
    def subscribe(self, event: str, handler: Callable) -> None
    def unsubscribe(self, event: str, handler: Callable) -> None
    def publish(self, event: str, **kwargs) -> None
    def clear(self, event: str | None = None) -> None

def get_bus() -> EventBus  # global singleton
```

---

## app.metrics_collector

```python
class Counter:
    def inc(self, amount: float = 1.0) -> None
    def reset(self) -> None
    value: float

class Gauge:
    def set(self, value: float) -> None
    def inc(self, amount: float = 1.0) -> None
    def dec(self, amount: float = 1.0) -> None
    value: float

class Histogram:
    def observe(self, value: float) -> None
    def percentile(self, p: float) -> float | None
    sum: float; count: int

class MetricsRegistry:
    def counter(self, name: str, description: str = "") -> Counter
    def gauge(self, name: str, description: str = "") -> Gauge
    def histogram(self, name: str, ...) -> Histogram

def get_registry() -> MetricsRegistry
```

---

## app.health_check

```python
class CheckResult:
    name: str; healthy: bool; message: str; details: dict

class HealthStatus:
    healthy: bool; results: List[CheckResult]
    failed: List[CheckResult]  # property

class HealthRegistry:
    def register(self, name: str, fn: Callable[[], CheckResult]) -> None
    def unregister(self, name: str) -> None
    def run(self) -> HealthStatus

def check(name: str, registry: HealthRegistry | None = None) -> Callable
```

---

## app.audit_log

```python
@dataclass(frozen=True)
class AuditEntry:
    actor: str; action: str; resource: str
    timestamp: float; metadata: dict; outcome: str

class AuditLog:
    def record(self, actor, action, resource, outcome="success", **metadata) -> AuditEntry
    def search(self, actor=None, action=None, ...) -> List[AuditEntry]
    def export_jsonl(self) -> str
```

---

## app.notification_dispatcher

```python
class Severity(str, Enum):
    INFO = "info"; WARNING = "warning"; ERROR = "error"; CRITICAL = "critical"

@dataclass
class Notification:
    title: str; body: str
    severity: Severity = Severity.INFO
    tags: List[str]

@dataclass
class Channel:
    name: str; send: Callable[[Notification], None]
    min_severity: Severity; enabled: bool

class NotificationDispatcher:
    def register(self, channel: Channel) -> None
    def dispatch(self, notification: Notification) -> Dict[str, bool]
    def set_enabled(self, name: str, enabled: bool) -> None
```

---

## app.geo_utils

```python
@dataclass(frozen=True)
class Coordinate:
    lat: float; lon: float

def haversine(a: Coordinate, b: Coordinate) -> float  # km

@dataclass
class BoundingBox:
    min_lat: float; max_lat: float; min_lon: float; max_lon: float
    def contains(self, coord: Coordinate) -> bool
    center: Coordinate  # property

def bounding_box_of(coords: List[Coordinate]) -> BoundingBox
def nearest_neighbor(target: Coordinate, candidates: List[Coordinate]) -> Coordinate
def midpoint(a: Coordinate, b: Coordinate) -> Coordinate
```
