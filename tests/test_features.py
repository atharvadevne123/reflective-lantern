"""Feature engineering pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def test_pipeline_handles_missing_values(sample_df) -> None:
    df = sample_df.copy()
    df.loc[0, "renovation_year"] = None
    pipeline = build_feature_pipeline()
    result = pipeline.fit_transform(df)
    non_optional = [c for c in result.columns if c != "renovation_year"]
    assert not result[non_optional].isnull().all().any()


@pytest.mark.parametrize(
    "bedrooms,bathrooms",
    [(1, 1.0), (2, 1.5), (4, 3.0), (6, 4.5)],
)
def test_beds_per_bath_various_configs(single_row, bedrooms, bathrooms) -> None:
    row = single_row.copy()
    row["bedrooms"] = bedrooms
    row["bathrooms"] = bathrooms
    result = RatioFeatureTransformer().fit_transform(row)
    assert result["beds_per_bath"].iloc[0] == pytest.approx(bedrooms / bathrooms)


@pytest.mark.parametrize("year_built", [1950, 1970, 1990, 2005, 2020])
def test_property_age_various_years(single_row, year_built) -> None:
    row = single_row.copy()
    row["year_built"] = year_built
    result = PropertyAgeTransformer(reference_year=2026).fit_transform(row)
    assert result["property_age"].iloc[0] == 2026 - year_built


def test_amenity_composite_high_scores(single_row) -> None:
    row = single_row.copy()
    row["school_score"] = 10.0
    row["transit_score"] = 10.0
    row["walkability_score"] = 10.0
    row["crime_rate"] = 0.0
    result = AmenityCompositeTransformer().fit_transform(row)
    assert result["amenity_composite"].iloc[0] > 0.8


def test_extract_feature_array_multiple_rows(sample_df) -> None:
    pipeline = build_feature_pipeline()
    arr = extract_feature_array(sample_df, pipeline)
    assert arr.ndim == 2
    assert arr.shape[0] == len(sample_df)
    assert np.all(np.isfinite(arr))


def test_school_weight_constant() -> None:
    from app.features import _SCHOOL_WEIGHT

    assert 0.0 < _SCHOOL_WEIGHT <= 1.0


def test_transit_weight_constant() -> None:
    from app.features import _TRANSIT_WEIGHT

    assert 0.0 < _TRANSIT_WEIGHT <= 1.0


def test_walk_weight_constant() -> None:
    from app.features import _WALK_WEIGHT

    assert 0.0 < _WALK_WEIGHT <= 1.0


def test_amenity_weights_sum_to_one() -> None:
    from app.features import _SCHOOL_WEIGHT, _TRANSIT_WEIGHT, _WALK_WEIGHT

    import pytest

    total = _SCHOOL_WEIGHT + _TRANSIT_WEIGHT + _WALK_WEIGHT
    assert total == pytest.approx(1.0)


def test_amenity_scale_constant() -> None:
    from app.features import _AMENITY_SCALE

    assert _AMENITY_SCALE > 0.0


@pytest.mark.parametrize("school,transit,walk", [
    (10.0, 10.0, 10.0),
    (5.0, 5.0, 5.0),
    (0.0, 0.0, 0.0),
])
def test_amenity_composite_uses_weights(school, transit, walk, single_row) -> None:
    from app.features import _AMENITY_SCALE, _SCHOOL_WEIGHT, _TRANSIT_WEIGHT, _WALK_WEIGHT, AmenityCompositeTransformer

    row = single_row.copy()
    row["school_score"] = school
    row["transit_score"] = transit
    row["walkability_score"] = walk
    result = AmenityCompositeTransformer().fit_transform(row)
    expected = (school * _SCHOOL_WEIGHT + transit * _TRANSIT_WEIGHT + walk * _WALK_WEIGHT) / _AMENITY_SCALE
    import pytest as _pytest
    assert result["amenity_composite"].iloc[0] == _pytest.approx(expected, rel=1e-3)
