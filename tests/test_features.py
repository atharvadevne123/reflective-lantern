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


def test_temporal_adds_cyclic_columns() -> None:
    df = _base_df()
    out = TemporalFeatureExtractor().fit_transform(df)
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_business_hour"):
        assert col in out.columns, f"Missing column: {col}"


def test_temporal_weekend_flag() -> None:
    df = pd.DataFrame({"hour": [10], "day_of_week": [6], "month": [1]})
    out = TemporalFeatureExtractor().fit_transform(df)
    assert out["is_weekend"].iloc[0] == 1


def test_temporal_weekday_flag() -> None:
    df = pd.DataFrame({"hour": [10], "day_of_week": [1], "month": [1]})
    out = TemporalFeatureExtractor().fit_transform(df)
    assert out["is_weekend"].iloc[0] == 0


def test_lag_extractor_adds_columns() -> None:
    df = _base_df()
    out = LagFeatureExtractor().fit_transform(df)
    for lag in [1, 2, 3, 6, 12, 24, 168]:
        assert f"lag_{lag}h" in out.columns


def test_lag_extractor_no_nans() -> None:
    df = _base_df()
    out = LagFeatureExtractor().fit_transform(df)
    for lag in [1, 2, 3]:
        assert out[f"lag_{lag}h"].isna().sum() == 0


def test_rolling_stats_columns() -> None:
    df = _base_df()
    out = RollingStatsExtractor().fit_transform(df)
    for w in [3, 6, 24]:
        for stat in ["mean", "std", "min", "max"]:
            assert f"roll_{stat}_{w}h" in out.columns


def test_weather_derived_features() -> None:
    df = _base_df()
    out = WeatherFeatureExtractor().fit_transform(df)
    for col in ("heat_index", "cooling_deg_hours", "heating_deg_hours", "temp_humidity_ratio"):
        assert col in out.columns


def test_weather_cooling_non_negative() -> None:
    df = pd.DataFrame({"temperature_c": [30.0], "humidity_pct": [60.0]})
    out = WeatherFeatureExtractor().fit_transform(df)
    assert out["cooling_deg_hours"].iloc[0] >= 0


def test_occupancy_occ_hvac_load() -> None:
    df = pd.DataFrame({"occupancy": [100], "hvac_state": [1]})
    out = OccupancyFeatureExtractor().fit_transform(df)
    assert out["occ_hvac_load"].iloc[0] == 100


def test_full_pipeline_output_shape() -> None:
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


def test_normalize_consumption_minmax_range() -> None:
    from app.features import normalize_consumption

    data = [2.0, 5.0, 8.0, 11.0]
    result = normalize_consumption(data, method="minmax")
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_normalize_consumption_zscore_mean() -> None:
    from app.features import normalize_consumption

    data = list(range(1, 11))
    result = normalize_consumption(data, method="zscore")
    assert abs(sum(result) / len(result)) < 1e-9


def test_normalize_consumption_flat_minmax() -> None:
    from app.features import normalize_consumption

    result = normalize_consumption([5.0] * 10, method="minmax")
    assert all(v == 0.0 for v in result)


def test_normalize_consumption_flat_zscore() -> None:
    from app.features import normalize_consumption

    result = normalize_consumption([3.0] * 8, method="zscore")
    assert all(v == 0.0 for v in result)


def test_normalize_consumption_empty_raises() -> None:
    from app.features import normalize_consumption

    with pytest.raises(ValueError, match="empty"):
        normalize_consumption([], method="minmax")


def test_normalize_consumption_bad_method_raises() -> None:
    from app.features import normalize_consumption

    with pytest.raises(ValueError, match="method"):
        normalize_consumption([1.0, 2.0], method="l2")


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_normalize_consumption_length_preserved(method) -> None:
    from app.features import normalize_consumption

    data = list(range(1, 21))
    result = normalize_consumption(data, method=method)
    assert len(result) == len(data)


def test_demand_response_potential_basic() -> None:
    from app.features import demand_response_potential

    loads = [10.0] * 20 + [100.0] * 4
    result = demand_response_potential(loads, peak_threshold_pct=0.85)
    assert result["peak_hours_count"] == 4
    assert result["sheddable_kwh"] >= 0


def test_demand_response_potential_flat_series() -> None:
    from app.features import demand_response_potential

    # For a flat series threshold = 0.85 * peak; values at peak all exceed it.
    # Sheddable per hour = peak - threshold = peak * (1 - 0.85) = 5.0 * 0.15 = 0.75
    result = demand_response_potential([5.0] * 24, peak_threshold_pct=0.85)
    assert result["peak_hours_count"] == 24
    assert result["sheddable_kwh"] == pytest.approx(24 * 0.75)


def test_demand_response_potential_empty_raises() -> None:
    from app.features import demand_response_potential

    with pytest.raises(ValueError):
        demand_response_potential([])


def test_demand_response_potential_bad_threshold_raises() -> None:
    from app.features import demand_response_potential

    with pytest.raises(ValueError):
        demand_response_potential([1.0, 2.0], peak_threshold_pct=0.0)


def test_demand_response_potential_keys() -> None:
    from app.features import demand_response_potential

    result = demand_response_potential([1.0, 5.0, 10.0], peak_threshold_pct=0.9)
    assert set(result.keys()) >= {"peak_hours_count", "sheddable_kwh", "potential_pct", "peak_threshold_kwh"}


def test_demand_response_potential_potential_pct_bounded() -> None:
    from app.features import demand_response_potential

    loads = list(range(1, 25))
    result = demand_response_potential(loads, peak_threshold_pct=0.5)
    assert 0.0 <= result["potential_pct"] <= 1.0


@pytest.mark.parametrize("threshold_pct", [0.5, 0.75, 0.9, 1.0])
def test_demand_response_potential_threshold_parametrized(threshold_pct) -> None:
    from app.features import demand_response_potential

    loads = [float(i) for i in range(1, 25)]
    result = demand_response_potential(loads, peak_threshold_pct=threshold_pct)
    assert result["peak_threshold_kwh"] == pytest.approx(max(loads) * threshold_pct, rel=1e-4)


def test_normalize_consumption_minmax() -> None:
    from app.features import normalize_consumption

    values = [0.0, 5.0, 10.0]
    result = normalize_consumption(values, method="minmax")
    assert abs(result[0] - 0.0) < 1e-9
    assert abs(result[-1] - 1.0) < 1e-9


def test_normalize_consumption_zscore() -> None:
    import statistics

    from app.features import normalize_consumption

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = normalize_consumption(values, method="zscore")
    assert abs(statistics.mean(result)) < 1e-6


def test_normalize_consumption_invalid_method() -> None:
    from app.features import normalize_consumption

    with pytest.raises(ValueError, match="method must be"):
        normalize_consumption([1.0, 2.0], method="unknown")


def test_normalize_consumption_empty() -> None:
    from app.features import normalize_consumption

    with pytest.raises(ValueError, match="must not be empty"):
        normalize_consumption([])


def test_normalize_consumption_constant_minmax() -> None:
    from app.features import normalize_consumption

    result = normalize_consumption([5.0] * 10, method="minmax")
    assert all(v == 0.0 for v in result)


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_normalize_consumption_same_length(method) -> None:
    from app.features import normalize_consumption

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = normalize_consumption(values, method=method)
    assert len(result) == len(values)


def test_demand_response_potential_extended() -> None:
    from app.features import demand_response_potential

    loads = [5.0, 10.0, 15.0, 20.0, 25.0]
    result = demand_response_potential(loads, peak_threshold_pct=0.8)
    assert "peak_hours_count" in result
    assert "sheddable_kwh" in result
    assert "potential_pct" in result
    assert result["peak_hours_count"] >= 1


def test_demand_response_potential_empty() -> None:
    from app.features import demand_response_potential

    with pytest.raises(ValueError, match="must not be empty"):
        demand_response_potential([])


def test_demand_response_potential_invalid_threshold() -> None:
    from app.features import demand_response_potential

    with pytest.raises(ValueError, match="peak_threshold_pct"):
        demand_response_potential([1.0, 2.0], peak_threshold_pct=0.0)


@pytest.mark.parametrize("threshold", [0.5, 0.8, 1.0])
def test_demand_response_potential_thresholds(threshold) -> None:
    from app.features import demand_response_potential

    loads = list(range(1, 25))
    result = demand_response_potential(loads, peak_threshold_pct=threshold)
    assert 0.0 <= result["potential_pct"] <= 1.0


def test_encode_cyclical_returns_two_floats() -> None:
    from app.features import encode_cyclical

    sin_val, cos_val = encode_cyclical(0.0, 24.0)
    assert isinstance(sin_val, float)
    assert isinstance(cos_val, float)


def test_encode_cyclical_zero_is_zero_sin() -> None:
    from app.features import encode_cyclical

    sin_val, _cos_val = encode_cyclical(0.0, 24.0)
    assert abs(sin_val) < 1e-9


def test_encode_cyclical_half_period_sine_near_zero() -> None:
    from app.features import encode_cyclical

    sin_val, _cos_val = encode_cyclical(12.0, 24.0)
    assert abs(sin_val) < 1e-9


def test_normalize_consumption_minmax_endpoints() -> None:
    from app.features import normalize_consumption

    result = normalize_consumption([0.0, 5.0, 10.0], method="minmax")
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_normalize_consumption_zscore_returns_list() -> None:
    from app.features import normalize_consumption

    result = normalize_consumption([10.0, 20.0, 30.0], method="zscore")
    assert isinstance(result, list)
    assert len(result) == 3


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_normalize_consumption_length_preserved_method(method) -> None:
    from app.features import normalize_consumption

    values = [float(i) for i in range(10)]
    result = normalize_consumption(values, method=method)
    assert len(result) == len(values)


class TestLagFeatures:
    def test_single_lag(self) -> None:
        from app.features import lag_features

        result = lag_features([1.0, 2.0, 3.0, 4.0], lags=[1])
        assert result["lag_1"] == [None, 1.0, 2.0, 3.0]

    def test_two_lags(self) -> None:
        from app.features import lag_features

        result = lag_features([10.0, 20.0, 30.0, 40.0], lags=[1, 2])
        assert "lag_1" in result and "lag_2" in result

    def test_invalid_lag(self) -> None:
        from app.features import lag_features

        with pytest.raises(ValueError):
            lag_features([1.0, 2.0], lags=[0])

    def test_length_preserved(self) -> None:
        from app.features import lag_features

        vals = [float(i) for i in range(10)]
        result = lag_features(vals, lags=[3])
        assert len(result["lag_3"]) == 10


class TestDifferenceFeature:
    def test_first_order(self) -> None:
        from app.features import difference_feature

        result = difference_feature([1.0, 3.0, 6.0, 10.0], order=1)
        assert result == pytest.approx([2.0, 3.0, 4.0])

    def test_second_order(self) -> None:
        from app.features import difference_feature

        result = difference_feature([1.0, 2.0, 4.0, 7.0], order=2)
        assert len(result) == 2

    def test_invalid_order(self) -> None:
        from app.features import difference_feature

        with pytest.raises(ValueError):
            difference_feature([1.0, 2.0], order=0)

    def test_constant_series(self) -> None:
        from app.features import difference_feature

        result = difference_feature([5.0, 5.0, 5.0], order=1)
        assert all(v == pytest.approx(0.0) for v in result)


class TestRatioFeature:
    def test_basic(self) -> None:
        from app.features import ratio_feature

        result = ratio_feature([10.0, 20.0], [2.0, 4.0])
        assert result == pytest.approx([5.0, 5.0])

    def test_zero_denominator(self) -> None:
        from app.features import ratio_feature

        result = ratio_feature([5.0], [0.0])
        assert result == [0.0]

    def test_empty_raises(self) -> None:
        from app.features import ratio_feature

        with pytest.raises(ValueError):
            ratio_feature([], [])

    def test_length_mismatch_raises(self) -> None:
        from app.features import ratio_feature

        with pytest.raises(ValueError):
            ratio_feature([1.0, 2.0], [1.0])


class TestClipFeatureValues:
    def test_clips_below(self) -> None:
        from app.features import clip_feature_values

        result = clip_feature_values([-5.0, 3.0], low=0.0, high=10.0)
        assert result[0] == pytest.approx(0.0)

    def test_clips_above(self) -> None:
        from app.features import clip_feature_values

        result = clip_feature_values([5.0, 20.0], low=0.0, high=10.0)
        assert result[1] == pytest.approx(10.0)

    def test_invalid_range_raises(self) -> None:
        from app.features import clip_feature_values

        with pytest.raises(ValueError):
            clip_feature_values([1.0], low=10.0, high=5.0)

    def test_empty(self) -> None:
        from app.features import clip_feature_values

        assert clip_feature_values([], low=0.0, high=1.0) == []


class TestBinFeature:
    def test_basic_binning(self) -> None:
        from app.features import bin_feature

        result = bin_feature([0.0, 5.0, 10.0, 15.0], bins=[5.0, 10.0])
        assert result == [0, 1, 2, 2]

    def test_below_all_bins(self) -> None:
        from app.features import bin_feature

        result = bin_feature([-1.0], bins=[0.0, 5.0])
        assert result == [0]

    def test_above_all_bins(self) -> None:
        from app.features import bin_feature

        result = bin_feature([100.0], bins=[0.0, 5.0, 10.0])
        assert result == [3]

    def test_empty_values_returns_empty(self) -> None:
        from app.features import bin_feature

        assert bin_feature([], bins=[1.0, 2.0]) == []

    def test_non_monotonic_bins_raises(self) -> None:
        from app.features import bin_feature

        with pytest.raises(ValueError):
            bin_feature([1.0], bins=[5.0, 2.0])

    @pytest.mark.parametrize(
        "val,expected",
        [
            (-1.0, 0),
            (2.5, 1),
            (7.5, 2),
            (15.0, 3),
        ],
    )
    def test_parametrize_bins(self, val: float, expected: int) -> None:
        from app.features import bin_feature

        result = bin_feature([val], bins=[0.0, 5.0, 10.0])
        assert result == [expected]


class TestCumulativeSumFeature:
    def test_empty_returns_empty(self) -> None:
        from app.features import cumulative_sum_feature

        assert cumulative_sum_feature([]) == []

    def test_ascending_cumsum(self) -> None:
        from app.features import cumulative_sum_feature

        result = cumulative_sum_feature([1.0, 2.0, 3.0])
        assert result == pytest.approx([1.0, 3.0, 6.0])

    def test_all_zeros(self) -> None:
        from app.features import cumulative_sum_feature

        assert cumulative_sum_feature([0.0, 0.0, 0.0]) == pytest.approx([0.0, 0.0, 0.0])

    def test_length_preserved(self) -> None:
        from app.features import cumulative_sum_feature

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(cumulative_sum_feature(values)) == len(values)


class TestRollingMaxFeature:
    def test_single_value(self) -> None:
        from app.features import rolling_max_feature

        assert rolling_max_feature([5.0]) == pytest.approx([5.0])

    def test_full_window_max(self) -> None:
        from app.features import rolling_max_feature

        result = rolling_max_feature([1.0, 3.0, 2.0, 4.0], window=3)
        assert result == pytest.approx([1.0, 3.0, 3.0, 4.0])

    def test_window_zero_raises(self) -> None:
        from app.features import rolling_max_feature

        with pytest.raises(ValueError):
            rolling_max_feature([1.0, 2.0], window=0)

    @pytest.mark.parametrize("window", [1, 2, 5])
    def test_length_preserved(self, window) -> None:
        from app.features import rolling_max_feature

        values = [float(i) for i in range(10)]
        assert len(rolling_max_feature(values, window=window)) == 10


class TestPercentileFeature:
    def test_empty_reference_raises(self) -> None:
        from app.features import percentile_feature

        with pytest.raises(ValueError):
            percentile_feature([1.0], reference=[])

    def test_invalid_percentile_raises(self) -> None:
        from app.features import percentile_feature

        with pytest.raises(ValueError):
            percentile_feature([1.0], reference=[1.0, 2.0], percentile=101.0)

    def test_all_above_median(self) -> None:
        from app.features import percentile_feature

        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = percentile_feature([3.0, 4.0, 5.0], ref, percentile=50.0)
        assert all(v == 1.0 for v in result)

    def test_returns_correct_length(self) -> None:
        from app.features import percentile_feature

        result = percentile_feature([1.0, 2.0, 3.0], reference=[1.0, 2.0, 3.0])
        assert len(result) == 3


@pytest.mark.parametrize("n", [5, 10, 24])
def test_cumulative_sum_feature_length(n: int) -> None:
    from app.features import cumulative_sum_feature

    values = [1.0] * n
    result = cumulative_sum_feature(values)
    assert len(result) == n


@pytest.mark.parametrize(
    "values,low,high",
    [
        ([1.0, 5.0, 10.0], 2.0, 8.0),
        ([-5.0, 0.0, 15.0], 0.0, 10.0),
    ],
)
def test_clip_feature_values_within_bounds(values: list, low: float, high: float) -> None:
    from app.features import clip_feature_values

    result = clip_feature_values(values, low, high)
    assert all(low <= v <= high for v in result)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_difference_feature_length_reduces(order: int) -> None:
    from app.features import difference_feature

    values = list(range(10))
    result = difference_feature(values, order=order)
    assert len(result) == 10 - order


@pytest.mark.parametrize(
    "value,max_value",
    [
        (0.0, 24.0),
        (6.0, 24.0),
        (12.0, 24.0),
    ],
)
def test_encode_cyclical_unit_circle(value: float, max_value: float) -> None:
    from app.features import encode_cyclical

    sin_val, cos_val = encode_cyclical(value, max_value)
    assert sin_val**2 + cos_val**2 == pytest.approx(1.0, abs=1e-6)


class TestRankFeatures:
    def test_sorted_descending(self) -> None:
        from app.features import rank_features

        imp = {"a": 0.1, "b": 0.5, "c": 0.3}
        result = rank_features(imp)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_dict_returns_empty(self) -> None:
        from app.features import rank_features

        assert rank_features({}) == []

    def test_all_names_present(self) -> None:
        from app.features import rank_features

        imp = {"x": 1.0, "y": 2.0}
        names = [n for n, _ in rank_features(imp)]
        assert set(names) == {"x", "y"}


class TestTopKFeatures:
    def test_returns_k_names(self) -> None:
        from app.features import top_k_features

        imp = {"a": 1.0, "b": 2.0, "c": 0.5, "d": 0.1}
        result = top_k_features(imp, k=2)
        assert len(result) == 2

    def test_highest_first(self) -> None:
        from app.features import top_k_features

        imp = {"a": 0.1, "b": 0.9, "c": 0.5}
        result = top_k_features(imp, k=1)
        assert result == ["b"]

    def test_k_larger_than_features(self) -> None:
        from app.features import top_k_features

        imp = {"a": 1.0, "b": 2.0}
        result = top_k_features(imp, k=10)
        assert len(result) == 2

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_returns_list_of_strings(self, k: int) -> None:
        from app.features import top_k_features

        imp = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = top_k_features(imp, k=k)
        assert all(isinstance(n, str) for n in result)
