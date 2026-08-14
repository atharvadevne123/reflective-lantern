"""Hotel booking feature engineering pipeline for Suite-Cast."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOM_TYPES: list[str] = ["standard", "deluxe", "suite"]
BOOKING_CHANNELS: list[str] = ["direct", "online", "ota"]

# Seasonal demand multiplier indexed by calendar month
_MONTH_TO_SEASON: dict[int, str] = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "fall",
    10: "fall",
    11: "fall",
}
_SEASON_INDEX: dict[str, float] = {
    "winter": 0.70,
    "spring": 1.00,
    "summer": 1.40,
    "fall": 0.90,
}

FEATURE_COLS: list[str] = [
    "lead_time",
    "length_of_stay",
    "guests_count",
    "checkin_month",
    "checkin_dayofweek",
    "is_weekend",
    "current_occ_rate",
    "prev_year_occ_rate",
    "room_rate",
    "special_event",
    # engineered
    "lead_time_bucket",
    "seasonality_score",
    "competitor_rate_ratio",
    "weekend_summer_flag",
    "yoy_occ_delta",
    "advance_efficiency",
    "room_type_enc",
    "channel_enc",
]


class HotelFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transform raw booking request fields into model-ready numeric features.

    Engineered features:
        lead_time_bucket: ordinal bin of lead-time (0=last-minute → 3=early).
        seasonality_score: demand multiplier derived from check-in month.
        competitor_rate_ratio: ratio of proposed rate to competitor average.
        weekend_summer_flag: interaction flag for weekend in peak summer.
        yoy_occ_delta: year-over-year occupancy rate difference (lag).
        advance_efficiency: lead_time × seasonality_score / 100 (interaction).
        room_type_enc, channel_enc: ordinal encodings of categorical fields.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> HotelFeatureEngineer:  # noqa: ANN001
        """No-op fit — all transforms are stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with engineered feature columns only."""
        df = X.copy()

        # Lag feature: ordinal bucket of booking lead time
        df["lead_time_bucket"] = pd.cut(
            df["lead_time"].clip(0, 365),
            bins=[-1, 3, 14, 60, 365],
            labels=[0, 1, 2, 3],
        ).astype(int)

        # Rolling / derived: seasonal demand index from check-in month
        df["season"] = df["checkin_month"].map(_MONTH_TO_SEASON).fillna("spring")
        df["seasonality_score"] = df["season"].map(_SEASON_INDEX).astype(float)

        # Ratio feature: our rate vs competitor average
        df["competitor_rate_ratio"] = (
            df["room_rate"] / df["competitor_avg_rate"].replace(0.0, 150.0)
        ).clip(0.5, 3.0)

        # Interaction: weekend AND summer peak flag
        df["is_summer"] = df["checkin_month"].isin([6, 7, 8]).astype(int)
        df["weekend_summer_flag"] = (
            df["is_weekend"].astype(bool) & df["is_summer"].astype(bool)
        ).astype(int)

        # Lag feature: year-over-year occupancy delta
        df["yoy_occ_delta"] = (df["current_occ_rate"] - df["prev_year_occ_rate"]).clip(-1.0, 1.0)

        # Ratio / interaction: advance booking efficiency
        df["advance_efficiency"] = (
            df["lead_time"].clip(1, 365).astype(float) * df["seasonality_score"]
        ) / 100.0

        # Ordinal encoding of categorical fields
        df["room_type_enc"] = pd.Categorical(df["room_type"], categories=ROOM_TYPES).codes
        df["channel_enc"] = pd.Categorical(df["booking_channel"], categories=BOOKING_CHANNELS).codes

        return df[FEATURE_COLS].astype(float)


def build_feature_pipeline() -> Pipeline:
    """Return a sklearn Pipeline combining feature engineering and scaling."""
    return Pipeline(
        [
            ("engineer", HotelFeatureEngineer()),
            ("scaler", StandardScaler()),
        ]
    )


def feature_names() -> list[str]:
    """Return the ordered list of feature column names output by the pipeline.

    Returns:
        List of feature column name strings as produced by HotelFeatureEngineer.
    """
    return list(FEATURE_COLS)


def lead_time_bucket_label(lead_time_days: int) -> str:
    """Return a human-readable label for a given lead time in days.

    Args:
        lead_time_days: Number of days between booking and check-in.

    Returns:
        One of 'last_minute', 'short', 'medium', 'advance'.
    """
    if lead_time_days <= 3:
        return "last_minute"
    if lead_time_days <= 14:
        return "short"
    if lead_time_days <= 60:
        return "medium"
    return "advance"


def seasonality_score_for_month(month: int) -> float:
    """Return the seasonal demand multiplier for a calendar month (1-12).

    Args:
        month: Calendar month as integer 1-12.

    Returns:
        Demand multiplier float from _SEASON_INDEX.
    """
    season = _MONTH_TO_SEASON.get(month, "spring")
    return _SEASON_INDEX[season]
