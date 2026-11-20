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


def is_valid_severity(label: str) -> bool:
    """Return True if label is a recognised severity level.

    Args:
        label: Severity string to validate.

    Returns:
        True when label is one of SEVERITY_LEVELS, False otherwise.
    """
    return label in SEVERITY_LEVELS


def severity_from_confidence(confidence: float) -> str:
    """Map a model confidence score to a severity label.

    Args:
        confidence: Probability in [0, 1] that an incident will occur.

    Returns:
        One of 'critical', 'high', 'medium', or 'low'.
    """
    if confidence >= SEVERITY_CRITICAL_THRESHOLD:
        return "critical"
    if confidence >= SEVERITY_HIGH_THRESHOLD:
        return "high"
    if confidence >= SEVERITY_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


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
    "is_valid_severity",
    "severity_from_confidence",
]
