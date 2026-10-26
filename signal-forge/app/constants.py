"""Shared constants for Signal-Forge."""

from __future__ import annotations

REGIMES: list[str] = ["bull", "bear", "sideways", "volatile"]

REGIME_RISK_WEIGHTS: dict[str, float] = {
    "bull": 0.2,
    "bear": 0.85,
    "sideways": 0.4,
    "volatile": 0.9,
}

RISK_TIERS: dict[str, tuple[float, float]] = {
    "low": (0.0, 0.25),
    "moderate": (0.25, 0.5),
    "high": (0.5, 0.75),
    "critical": (0.75, 1.0),
}

FEATURE_NAMES: list[str] = [
    "volatility",
    "momentum",
    "volume_ratio",
    "correlation",
    "beta",
]

ANNUALISATION_FACTOR: float = 252.0 ** 0.5
DEFAULT_ROLLING_WINDOW: int = 21
BETA_ROLLING_WINDOW: int = 63
MIN_DRIFT_SAMPLES: int = 10
FAISS_DEFAULT_K: int = 3
MODEL_VERSION: str = "1.0.0"
