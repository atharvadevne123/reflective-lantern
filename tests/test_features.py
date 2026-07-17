"""Feature engineering pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    AmenityCompositeTransformer,
    LagFeatureExtractor,
    OccupancyFeatureExtractor,
    PropertyAgeTransformer,
    RatioFeatureTransformer,
    RollingStatsExtractor,
    TemporalFeatureExtractor,
    WeatherFeatureExtractor,
    build_feature_pipeline,
    extract_feature_array,
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
    import pytest

    from app.features import _SCHOOL_WEIGHT, _TRANSIT_WEIGHT, _WALK_WEIGHT

    total = _SCHOOL_WEIGHT + _TRANSIT_WEIGHT + _WALK_WEIGHT
    assert total == pytest.approx(1.0)


def test_amenity_scale_constant() -> None:
    from app.features import _AMENITY_SCALE

    assert _AMENITY_SCALE > 0.0


@pytest.mark.parametrize(
    "school,transit,walk",
    [
        (10.0, 10.0, 10.0),
        (5.0, 5.0, 5.0),
        (0.0, 0.0, 0.0),
    ],
)
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


def test_interaction_extractor_creates_columns() -> None:
    from app.features import InteractionFeatureExtractor

    df = _base_df(10)
    tx = InteractionFeatureExtractor()
    out = tx.fit_transform(df)
    assert "temperature_c_x_occupancy" in out.columns


def test_interaction_extractor_values_correct() -> None:
    from app.features import InteractionFeatureExtractor

    df = _base_df(5)
    out = InteractionFeatureExtractor().fit_transform(df)
    expected = df["temperature_c"] * df["occupancy"]
    assert (out["temperature_c_x_occupancy"].values == expected.values).all()


def test_interaction_extractor_missing_column_skipped() -> None:
    from app.features import InteractionFeatureExtractor

    df = pd.DataFrame({"temperature_c": [20.0, 25.0]})
    out = InteractionFeatureExtractor().fit_transform(df)
    assert "temperature_c_x_occupancy" not in out.columns


def test_interaction_extractor_is_stateless() -> None:
    from app.features import InteractionFeatureExtractor

    tx = InteractionFeatureExtractor()
    df = _base_df(4)
    tx.fit(df)
    assert hasattr(tx, "available_pairs_")


def test_normalize_consumption_minmax_range():
    from app.features import normalize_consumption

    data = [2.0, 5.0, 8.0, 11.0]
    result = normalize_consumption(data, method="minmax")
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_normalize_consumption_zscore_mean():
    from app.features import normalize_consumption

    data = list(range(1, 11))
    result = normalize_consumption(data, method="zscore")
    assert abs(sum(result) / len(result)) < 1e-9


def test_normalize_consumption_flat_minmax():
    from app.features import normalize_consumption

    result = normalize_consumption([5.0] * 10, method="minmax")
    assert all(v == 0.0 for v in result)


def test_normalize_consumption_flat_zscore():
    from app.features import normalize_consumption

    result = normalize_consumption([3.0] * 8, method="zscore")
    assert all(v == 0.0 for v in result)


def test_normalize_consumption_empty_raises():
    from app.features import normalize_consumption

    with pytest.raises(ValueError, match="empty"):
        normalize_consumption([], method="minmax")


def test_normalize_consumption_bad_method_raises():
    from app.features import normalize_consumption

    with pytest.raises(ValueError, match="method"):
        normalize_consumption([1.0, 2.0], method="l2")


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_normalize_consumption_length_preserved(method):
    from app.features import normalize_consumption

    data = list(range(1, 21))
    result = normalize_consumption(data, method=method)
    assert len(result) == len(data)


def test_demand_response_potential_basic():
    from app.features import demand_response_potential

    loads = [10.0] * 20 + [100.0] * 4
    result = demand_response_potential(loads, peak_threshold_pct=0.85)
    assert result["peak_hours_count"] == 4
    assert result["sheddable_kwh"] >= 0


def test_demand_response_potential_flat_series():
    from app.features import demand_response_potential

    # For a flat series threshold = 0.85 * peak; values at peak all exceed it.
    # Sheddable per hour = peak - threshold = peak * (1 - 0.85) = 5.0 * 0.15 = 0.75
    result = demand_response_potential([5.0] * 24, peak_threshold_pct=0.85)
    assert result["peak_hours_count"] == 24
    assert result["sheddable_kwh"] == pytest.approx(24 * 0.75)


def test_demand_response_potential_empty_raises():
    from app.features import demand_response_potential

    with pytest.raises(ValueError):
        demand_response_potential([])


def test_demand_response_potential_bad_threshold_raises():
    from app.features import demand_response_potential

    with pytest.raises(ValueError):
        demand_response_potential([1.0, 2.0], peak_threshold_pct=0.0)


def test_demand_response_potential_keys():
    from app.features import demand_response_potential

    result = demand_response_potential([1.0, 5.0, 10.0], peak_threshold_pct=0.9)
    assert set(result.keys()) >= {"peak_hours_count", "sheddable_kwh", "potential_pct", "peak_threshold_kwh"}


def test_demand_response_potential_potential_pct_bounded():
    from app.features import demand_response_potential

    loads = list(range(1, 25))
    result = demand_response_potential(loads, peak_threshold_pct=0.5)
    assert 0.0 <= result["potential_pct"] <= 1.0


@pytest.mark.parametrize("threshold_pct", [0.5, 0.75, 0.9, 1.0])
def test_demand_response_potential_threshold_parametrized(threshold_pct):
    from app.features import demand_response_potential

    loads = [float(i) for i in range(1, 25)]
    result = demand_response_potential(loads, peak_threshold_pct=threshold_pct)
    assert result["peak_threshold_kwh"] == pytest.approx(max(loads) * threshold_pct, rel=1e-4)


def test_normalize_consumption_minmax():
    from app.features import normalize_consumption
    values = [0.0, 5.0, 10.0]
    result = normalize_consumption(values, method="minmax")
    assert abs(result[0] - 0.0) < 1e-9
    assert abs(result[-1] - 1.0) < 1e-9


def test_normalize_consumption_zscore():
    from app.features import normalize_consumption
    import statistics
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = normalize_consumption(values, method="zscore")
    assert abs(statistics.mean(result)) < 1e-6


def test_normalize_consumption_invalid_method():
    from app.features import normalize_consumption
    with pytest.raises(ValueError, match="method must be"):
        normalize_consumption([1.0, 2.0], method="unknown")


def test_normalize_consumption_empty():
    from app.features import normalize_consumption
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_consumption([])


def test_normalize_consumption_constant_minmax():
    from app.features import normalize_consumption
    result = normalize_consumption([5.0] * 10, method="minmax")
    assert all(v == 0.0 for v in result)


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_normalize_consumption_same_length(method):
    from app.features import normalize_consumption
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = normalize_consumption(values, method=method)
    assert len(result) == len(values)


def test_demand_response_potential_basic():
    from app.features import demand_response_potential
    loads = [5.0, 10.0, 15.0, 20.0, 25.0]
    result = demand_response_potential(loads, peak_threshold_pct=0.8)
    assert "peak_hours_count" in result
    assert "sheddable_kwh" in result
    assert "potential_pct" in result
    assert result["peak_hours_count"] >= 1


def test_demand_response_potential_empty():
    from app.features import demand_response_potential
    with pytest.raises(ValueError, match="must not be empty"):
        demand_response_potential([])


def test_demand_response_potential_invalid_threshold():
    from app.features import demand_response_potential
    with pytest.raises(ValueError, match="peak_threshold_pct"):
        demand_response_potential([1.0, 2.0], peak_threshold_pct=0.0)


@pytest.mark.parametrize("threshold", [0.5, 0.8, 1.0])
def test_demand_response_potential_thresholds(threshold):
    from app.features import demand_response_potential
    loads = list(range(1, 25))
    result = demand_response_potential(loads, peak_threshold_pct=threshold)
    assert 0.0 <= result["potential_pct"] <= 1.0
