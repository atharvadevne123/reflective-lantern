"""Shared constants for the Watt-Guard application."""

from __future__ import annotations

# ── Anomaly Detection ──────────────────────────────────────────────────────────
ZSCORE_THRESHOLD: float = 2.5
IQR_MULTIPLIER: float = 1.5
MIN_REFERENCE_SIZE: int = 4

# ── Drift Monitoring ──────────────────────────────────────────────────────────
REFERENCE_WINDOW_SIZE: int = 500
DRIFT_P_THRESHOLD: float = 0.05

# ── Energy / Building ─────────────────────────────────────────────────────────
MAX_CONSUMPTION_KWH: float = 10_000.0
MIN_CONSUMPTION_KWH: float = 0.0
MAX_TEMPERATURE_C: float = 60.0
MIN_TEMPERATURE_C: float = -50.0
MAX_HUMIDITY_PCT: float = 100.0
MIN_HUMIDITY_PCT: float = 0.0
MAX_OCCUPANCY: int = 10_000
MAX_FORECAST_HORIZON: int = 168  # hours (one week)

# ── Carbon Emission Factors (kg CO2 per kWh) ──────────────────────────────────
GRID_INTENSITY_KG_PER_KWH: dict[str, float] = {
    "northeast": 0.37,
    "midwest": 0.51,
    "south": 0.43,
    "west": 0.23,
    "texas": 0.40,
    "pacific_nw": 0.10,
    "new_england": 0.28,
    "mountain": 0.45,
    "southeast": 0.44,
    "florida": 0.42,
    "default": 0.40,
}

# ── Cache ─────────────────────────────────────────────────────────────────────
DEFAULT_CACHE_TTL_SECONDS: int = 30
DEFAULT_CACHE_MAX_SIZE: int = 500

# ── API Rate Limiting ─────────────────────────────────────────────────────────
DEFAULT_RATE_LIMIT_PER_MINUTE: int = 120

# ── Reporting Thresholds ──────────────────────────────────────────────────────
EFFICIENCY_GRADE_A_THRESHOLD: float = 0.80
EFFICIENCY_GRADE_B_THRESHOLD: float = 0.60
EFFICIENCY_GRADE_C_THRESHOLD: float = 0.40

# ── HTTP / API Defaults ──────────────────────────────────────────────────────
DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_BATCH_SIZE: int = 100
DEFAULT_PAGE_SIZE: int = 50
MAX_PAGE_SIZE: int = 500

__all__ = [
    "DEFAULT_CACHE_MAX_SIZE",
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_MAX_BATCH_SIZE",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_RATE_LIMIT_PER_MINUTE",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DRIFT_P_THRESHOLD",
    "EFFICIENCY_GRADE_A_THRESHOLD",
    "EFFICIENCY_GRADE_B_THRESHOLD",
    "EFFICIENCY_GRADE_C_THRESHOLD",
    "GRID_INTENSITY_KG_PER_KWH",
    "IQR_MULTIPLIER",
    "MAX_CONSUMPTION_KWH",
    "MAX_FORECAST_HORIZON",
    "MAX_HUMIDITY_PCT",
    "MAX_OCCUPANCY",
    "MAX_PAGE_SIZE",
    "MAX_TEMPERATURE_C",
    "MIN_CONSUMPTION_KWH",
    "MIN_HUMIDITY_PCT",
    "MIN_REFERENCE_SIZE",
    "MIN_TEMPERATURE_C",
    "REFERENCE_WINDOW_SIZE",
    "ZSCORE_THRESHOLD",
]
