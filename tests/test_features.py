"""Feature engineering pipeline tests."""

from __future__ import annotations

from __future__ import annotations

import pytest

from app.features import (
    LagFeatureExtractor,
    OccupancyFeatureExtractor,
    RollingStatsExtractor,
    TemporalFeatureExtractor,
    WeatherFeatureExtractor,
    build_feature_pipeline,
    make_feature_row,
)


def _base_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "month": rng.integers(1, 13, n),
            "temperature_c": rng.uniform(0, 40, n),
            "humidity_pct": rng.uniform(20, 90, n),
            "occupancy": rng.integers(0, 200, n),
            "hvac_state": rng.integers(0, 2, n),
            "consumption_kwh": rng.uniform(5, 30, n),
        }
    )


def test_temporal_adds_cyclic_columns():
    df = _base_df()
    out = TemporalFeatureExtractor().fit_transform(df)
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_business_hour"):
        assert col in out.columns, f"Missing column: {col}"


def test_temporal_weekend_flag():
    df = pd.DataFrame({"hour": [10], "day_of_week": [6], "month": [1]})
    out = TemporalFeatureExtractor().fit_transform(df)
    assert out["is_weekend"].iloc[0] == 1


def test_temporal_weekday_flag():
    df = pd.DataFrame({"hour": [10], "day_of_week": [1], "month": [1]})
    out = TemporalFeatureExtractor().fit_transform(df)
    assert out["is_weekend"].iloc[0] == 0


def test_lag_extractor_adds_columns():
    df = _base_df()
    out = LagFeatureExtractor().fit_transform(df)
    for lag in [1, 2, 3, 6, 12, 24, 168]:
        assert f"lag_{lag}h" in out.columns


def test_lag_extractor_no_nans():
    df = _base_df()
    out = LagFeatureExtractor().fit_transform(df)
    for lag in [1, 2, 3]:
        assert out[f"lag_{lag}h"].isna().sum() == 0


def test_rolling_stats_columns():
    df = _base_df()
    out = RollingStatsExtractor().fit_transform(df)
    for w in [3, 6, 24]:
        for stat in ["mean", "std", "min", "max"]:
            assert f"roll_{stat}_{w}h" in out.columns


def test_weather_derived_features():
    df = _base_df()
    out = WeatherFeatureExtractor().fit_transform(df)
    for col in ("heat_index", "cooling_deg_hours", "heating_deg_hours", "temp_humidity_ratio"):
        assert col in out.columns


def test_weather_cooling_non_negative():
    df = pd.DataFrame({"temperature_c": [30.0], "humidity_pct": [60.0]})
    out = WeatherFeatureExtractor().fit_transform(df)
    assert out["cooling_deg_hours"].iloc[0] >= 0


def test_occupancy_occ_hvac_load():
    df = pd.DataFrame({"occupancy": [100], "hvac_state": [1]})
    out = OccupancyFeatureExtractor().fit_transform(df)
    assert out["occ_hvac_load"].iloc[0] == 100


def test_full_pipeline_output_shape():
    df = _base_df(100)
    pipe = build_feature_pipeline()
    result = pipe.fit_transform(df)
    assert result.shape[0] == 100
    assert result.shape[1] > 10


def test_full_pipeline_no_nan():
    df = _base_df(50)
    pipe = build_feature_pipeline()
    result = pipe.fit_transform(df)
    assert not np.isnan(result).any()


def test_make_feature_row_single():
    row = make_feature_row(14, 1, 6, 28.5, 60.0, 50, 1, 12.0)
    assert len(row) == 1
    assert row["hour"].iloc[0] == 14


@pytest.mark.parametrize("hour,expected_biz", [(9, 1), (22, 0), (8, 1), (18, 1), (19, 0)])
def test_business_hour_flag(hour, expected_biz):
    df = pd.DataFrame({"hour": [hour], "day_of_week": [1], "month": [3]})
    out = TemporalFeatureExtractor().fit_transform(df)
    assert out["is_business_hour"].iloc[0] == expected_biz


def test_pipeline_transform_matches_fit_transform():
    df = _base_df(60)
    pipe = build_feature_pipeline()
    r1 = pipe.fit_transform(df)
    r2 = pipe.transform(df)
    np.testing.assert_allclose(r1, r2, rtol=1e-5)
