"""Shared application-level constants for Ops-Vision."""

APP_NAME: str = "Ops-Vision"
APP_DESCRIPTION: str = (
    "SRE ML platform for real-time incident prediction, "
    "alert classification, and performance anomaly detection."
)

SEVERITY_CRITICAL_THRESHOLD: float = 0.9
SEVERITY_HIGH_THRESHOLD: float = 0.75
SEVERITY_MEDIUM_THRESHOLD: float = 0.5

SEVERITY_LEVELS: tuple[str, ...] = ("low", "medium", "high", "critical")

MAX_BATCH_SIZE: int = 100
MAX_TOP_K: int = 10
MIN_QUERY_LENGTH: int = 3

HEALTH_EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}
)

DEFAULT_PAGINATION_LIMIT: int = 50
MAX_PAGINATION_LIMIT: int = 500

__all__ = [
    "APP_DESCRIPTION",
    "APP_NAME",
    "DEFAULT_PAGINATION_LIMIT",
    "HEALTH_EXEMPT_PATHS",
    "MAX_BATCH_SIZE",
    "MAX_PAGINATION_LIMIT",
    "MAX_TOP_K",
    "MIN_QUERY_LENGTH",
    "SEVERITY_CRITICAL_THRESHOLD",
    "SEVERITY_HIGH_THRESHOLD",
    "SEVERITY_LEVELS",
    "SEVERITY_MEDIUM_THRESHOLD",
]
