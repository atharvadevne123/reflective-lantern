"""Extended suite-cast model tests — second pass with parametrize."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.model import (
    HIGH_DEMAND_THRESHOLD,
    MEDIUM_DEMAND_THRESHOLD,
    PRICE_FLOOR,
    PRICE_SPAN,
    generate_sample_data,
)


def test_generate_sample_data_returns_tuple() -> None:
    result = generate_sample_data(n=50)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_generate_sample_data_df_shape() -> None:
    df, y = generate_sample_data(n=100)
    assert len(df) == 100
    assert len(y) == 100


def test_generate_sample_data_df_has_required_columns() -> None:
    df, _ = generate_sample_data(n=50)
    required = [
        "lead_time",
        "length_of_stay",
        "guests_count",
        "checkin_month",
        "checkin_dayofweek",
        "room_rate",
        "competitor_avg_rate",
    ]
    for col in required:
        assert col in df.columns


def test_generate_sample_data_binary_target() -> None:
    _, y = generate_sample_data(n=100)
    assert set(np.unique(y)).issubset({0, 1})


def test_generate_sample_data_lead_time_non_negative() -> None:
    df, _ = generate_sample_data(n=200)
    assert (df["lead_time"] >= 0).all()


def test_generate_sample_data_room_rate_positive() -> None:
    df, _ = generate_sample_data(n=100)
    assert (df["room_rate"] > 0).all()


def test_generate_sample_data_checkin_month_in_range() -> None:
    df, _ = generate_sample_data(n=200)
    assert df["checkin_month"].between(1, 12).all()


@pytest.mark.parametrize("n", [10, 50, 200, 500])
def test_generate_sample_data_various_sizes(n: int) -> None:
    df, y = generate_sample_data(n=n)
    assert len(df) == n
    assert len(y) == n


def test_generate_sample_data_deterministic() -> None:
    df1, y1 = generate_sample_data(n=100)
    df2, y2 = generate_sample_data(n=100)
    pd.testing.assert_frame_equal(df1, df2)
    np.testing.assert_array_equal(y1, y2)


def test_price_floor_less_than_one() -> None:
    assert 0.0 < PRICE_FLOOR < 1.0


def test_price_span_positive() -> None:
    assert PRICE_SPAN > 0.0


def test_high_demand_threshold_above_medium() -> None:
    assert HIGH_DEMAND_THRESHOLD > MEDIUM_DEMAND_THRESHOLD


def test_high_demand_threshold_in_unit_interval() -> None:
    assert 0.0 < HIGH_DEMAND_THRESHOLD < 1.0


def test_medium_demand_threshold_in_unit_interval() -> None:
    assert 0.0 < MEDIUM_DEMAND_THRESHOLD < 1.0


def test_generate_sample_data_occ_rate_clamped() -> None:
    df, _ = generate_sample_data(n=200)
    assert (df["current_occ_rate"] >= 0.0).all()
    assert (df["current_occ_rate"] <= 1.0).all()
