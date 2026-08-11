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


def lead_time_bucket(lead_days: int) -> int:
    """Return an ordinal bucket for booking lead time.

    Buckets:
        0 — same-day or next 3 days  (0–3)
        1 — early (4–14 days)
        2 — planned (15–60 days)
        3 — advance (>60 days)

    Args:
        lead_days: Number of days between booking and check-in; clamped to [0, 365].

    Returns:
        Integer bucket label 0–3.
    """
    clamped = max(0, min(365, lead_days))
    if clamped <= 3:
        return 0
    if clamped <= 14:
        return 1
    if clamped <= 60:
        return 2
    return 3


def competitor_rate_ratio(room_rate: float, competitor_avg: float) -> float:
    """Return the ratio of *room_rate* to *competitor_avg*, clamped to [0.5, 3.0].

    Args:
        room_rate: Proposed room rate.
        competitor_avg: Competitor average rate; replaced with 150.0 if zero.

    Returns:
        Ratio clamped to [0.5, 3.0].
    """
    denom = competitor_avg if competitor_avg != 0.0 else 150.0
    return max(0.5, min(3.0, room_rate / denom))


def occupancy_yoy_delta(current: float, prev_year: float) -> float:
    """Return year-over-year occupancy rate difference, clamped to [-1, 1].

    Args:
        current: Current period occupancy rate.
        prev_year: Previous year's occupancy rate for the same period.

    Returns:
        Delta clamped to [-1, 1].
    """
    return max(-1.0, min(1.0, current - prev_year))


def adr_from_revenue(total_revenue: float, rooms_sold: int) -> float:
    """Return Average Daily Rate (ADR) from revenue and rooms sold.

    Args:
        total_revenue: Total room revenue for the period.
        rooms_sold: Number of rooms sold; must be positive.

    Returns:
        ADR as float; 0.0 if rooms_sold is zero.
    """
    if rooms_sold <= 0:
        return 0.0
    return round(total_revenue / rooms_sold, 4)


def revpar(total_revenue: float, available_rooms: int) -> float:
    """Return Revenue Per Available Room (RevPAR).

    Args:
        total_revenue: Total room revenue for the period.
        available_rooms: Total available room-nights; must be positive.

    Returns:
        RevPAR as float; 0.0 if available_rooms is zero.
    """
    if available_rooms <= 0:
        return 0.0
    return round(total_revenue / available_rooms, 4)


def length_of_stay_bucket(nights: int) -> str:
    """Categorise a booking by length of stay.

    Args:
        nights: Number of nights in the booking.

    Returns:
        'short' (1-2), 'medium' (3-6), or 'long' (7+).
    """
    if nights <= 2:
        return "short"
    if nights <= 6:
        return "medium"
    return "long"
