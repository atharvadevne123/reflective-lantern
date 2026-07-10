"""Feature engineering pipeline for traffic congestion prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROAD_TYPE_MAP: dict[str, int] = {
    "highway": 0,
    "arterial": 1,
    "collector": 2,
    "local": 3,
    "expressway": 4,
}

PEAK_HOURS: frozenset[int] = frozenset(range(7, 10)) | frozenset(range(16, 20))

FEATURE_COLUMNS: list[str] = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "vehicle_count",
    "avg_speed_kmh",
    "speed_volume_ratio",
    "incident_count",
    "incident_density",
    "temperature_celsius",
    "is_raining",
    "road_type_encoded",
    "lag_1h",
    "lag_2h",
    "lag_4h",
    "lag_delta_1h",
    "lag_ratio_1h",
    "rolling_mean_6h",
    "rolling_std_6h",
    "rolling_mean_24h",
    "volume_vs_daily_avg",
]


def encode_road_type(road_type: str) -> int:
    """Map a road type name to its ordinal encoding.

    Args:
        road_type: Road classification (case-insensitive).

    Returns:
        Integer code; unknown types fall back to ``local`` (3).
    """
    return ROAD_TYPE_MAP.get(road_type.lower(), 3)


def is_peak_hour(hour: int) -> int:
    """Return 1 when the hour falls in the 7-9 or 16-19 rush windows."""
    return int(hour in PEAK_HOURS)


def compute_lag_features(
    vehicle_count: float,
    avg_speed: float,  # noqa: ARG001
    lag_1h: float | None = None,
    lag_2h: float | None = None,
    lag_4h: float | None = None,
) -> dict[str, float]:
    """Compute lag features, falling back to decayed estimates when history is absent.

    Args:
        vehicle_count: Current hourly vehicle count.
        avg_speed: Current average speed (reserved for future lag interactions).
        lag_1h: Observed count one hour ago, if available.
        lag_2h: Observed count two hours ago, if available.
        lag_4h: Observed count four hours ago, if available.

    Returns:
        Lag values plus delta and ratio against the 1-hour lag.
    """
    l1 = lag_1h if lag_1h is not None else vehicle_count * 0.95
    l2 = lag_2h if lag_2h is not None else vehicle_count * 0.90
    l4 = lag_4h if lag_4h is not None else vehicle_count * 0.85
    return {
        "lag_1h": l1,
        "lag_2h": l2,
        "lag_4h": l4,
        "lag_delta_1h": vehicle_count - l1,
        "lag_ratio_1h": vehicle_count / (l1 + 1e-8),
    }


def compute_rolling_features(
    vehicle_count: float,
    rolling_mean_6h: float | None = None,
    rolling_std_6h: float | None = None,
    rolling_mean_24h: float | None = None,
) -> dict[str, float]:
    """Compute rolling-window statistics with sensible fallbacks.

    Returns:
        6h/24h rolling means, 6h std, and current volume vs daily average.
    """
    rm6 = rolling_mean_6h if rolling_mean_6h is not None else vehicle_count
    rs6 = rolling_std_6h if rolling_std_6h is not None else vehicle_count * 0.1
    rm24 = rolling_mean_24h if rolling_mean_24h is not None else vehicle_count * 0.95
    return {
        "rolling_mean_6h": rm6,
        "rolling_std_6h": rs6,
        "rolling_mean_24h": rm24,
        "volume_vs_daily_avg": vehicle_count / (rm24 + 1e-8),
    }


def build_feature_vector(raw: dict[str, Any]) -> dict[str, float]:
    """Convert a raw input dict into the flat feature vector consumed by the model."""
    hour = int(raw["hour"])
    dow = int(raw["day_of_week"])
    month = int(raw.get("month", 6))
    vehicle_count = float(raw.get("vehicle_count", 1000))
    avg_speed = float(raw.get("avg_speed_kmh", 50.0))
    incident_count = int(raw.get("incident_count", 0))
    temperature = float(raw.get("temperature_celsius", 20.0))
    is_raining = int(raw.get("is_raining", 0))
    road_type = str(raw.get("road_type", "arterial"))

    lag = compute_lag_features(
        vehicle_count,
        avg_speed,
        raw.get("lag_1h"),
        raw.get("lag_2h"),
        raw.get("lag_4h"),
    )
    rolling = compute_rolling_features(
        vehicle_count,
        raw.get("rolling_mean_6h"),
        raw.get("rolling_std_6h"),
        raw.get("rolling_mean_24h"),
    )

    return {
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "is_weekend": int(dow >= 5),
        "is_peak_hour": is_peak_hour(hour),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        "vehicle_count": vehicle_count,
        "avg_speed_kmh": avg_speed,
        "speed_volume_ratio": avg_speed / (vehicle_count + 1e-8),
        "incident_count": float(incident_count),
        "incident_density": incident_count / (vehicle_count + 1e-8),
        "temperature_celsius": temperature,
        "is_raining": float(is_raining),
        "road_type_encoded": float(encode_road_type(road_type)),
        **lag,
        **rolling,
    }


def build_feature_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Vectorise a batch of raw records into a model-ready dataframe."""
    rows = [build_feature_vector(r) for r in records]
    df = pd.DataFrame(rows)
    logger.info("Built feature dataframe shape=%s", df.shape)
    return df
