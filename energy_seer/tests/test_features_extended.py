"""Extended tests for energy_seer app/features.py."""

from __future__ import annotations

import pandas as pd
import pytest

from app.features import (
    BuildingEncoderTransformer,
    TemporalFeatureTransformer,
    describe_dataframe,
    validate_dataframe_columns,
)


class TestValidateDataframeColumnsExtended:
    def test_all_present_returns_empty(self) -> None:
        df = pd.DataFrame({"consumption_kwh": [1.0], "hour_of_day": [0], "day_of_week": [1]})
        missing = validate_dataframe_columns(df)
        assert missing == []

    def test_missing_column_reported(self) -> None:
        df = pd.DataFrame({"hour_of_day": [0]})
        missing = validate_dataframe_columns(df, required=["consumption_kwh", "hour_of_day"])
        assert "consumption_kwh" in missing

    def test_empty_dataframe_all_missing(self) -> None:
        df = pd.DataFrame()
        missing = validate_dataframe_columns(df, required=["a", "b"])
        assert set(missing) == {"a", "b"}

    def test_custom_required_list(self) -> None:
        df = pd.DataFrame({"x": [1], "y": [2]})
        missing = validate_dataframe_columns(df, required=["x", "z"])
        assert "z" in missing
        assert "x" not in missing

    def test_no_required_uses_defaults(self) -> None:
        df = pd.DataFrame({"consumption_kwh": [1.0], "hour_of_day": [0], "day_of_week": [1]})
        missing = validate_dataframe_columns(df)
        assert missing == []


class TestDescribeDataframeExtended:
    def test_empty_dataframe_returns_empty(self) -> None:
        df = pd.DataFrame()
        result = describe_dataframe(df)
        assert result == {}

    def test_numeric_columns_summarised(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = describe_dataframe(df)
        assert "a" in result
        assert "b" in result

    def test_mean_is_correct(self) -> None:
        df = pd.DataFrame({"vals": [2.0, 4.0, 6.0]})
        result = describe_dataframe(df)
        assert result["vals"]["mean"] == pytest.approx(4.0)

    def test_min_max_correct(self) -> None:
        df = pd.DataFrame({"vals": [1.0, 5.0, 3.0]})
        result = describe_dataframe(df)
        assert result["vals"]["min"] == pytest.approx(1.0)
        assert result["vals"]["max"] == pytest.approx(5.0)

    def test_string_columns_excluded(self) -> None:
        df = pd.DataFrame({"name": ["a", "b"], "value": [1.0, 2.0]})
        result = describe_dataframe(df)
        assert "name" not in result
        assert "value" in result

    def test_single_value_std_zero(self) -> None:
        df = pd.DataFrame({"x": [42.0]})
        result = describe_dataframe(df)
        assert result["x"]["std"] == pytest.approx(0.0)

    @pytest.mark.parametrize("n", [2, 5, 10, 50])
    def test_various_lengths(self, n: int) -> None:
        df = pd.DataFrame({"vals": [float(i) for i in range(n)]})
        result = describe_dataframe(df)
        assert "vals" in result
        assert result["vals"]["min"] == pytest.approx(0.0)
        assert result["vals"]["max"] == pytest.approx(float(n - 1))


class TestTemporalFeatureTransformerExtended:
    def test_adds_hour_sin_cos(self) -> None:
        df = pd.DataFrame({"hour_of_day": [0, 6, 12, 18]})
        result = TemporalFeatureTransformer().transform(df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_adds_peak_flags(self) -> None:
        df = pd.DataFrame({"hour_of_day": [8]})
        result = TemporalFeatureTransformer().transform(df)
        assert "is_peak_morning" in result.columns
        assert result["is_peak_morning"].iloc[0] == 1

    def test_no_hour_column_no_hour_features(self) -> None:
        df = pd.DataFrame({"day_of_week": [1, 2, 3]})
        result = TemporalFeatureTransformer().transform(df)
        assert "hour_sin" not in result.columns

    def test_adds_weekend_flag(self) -> None:
        df = pd.DataFrame({"day_of_week": [5, 6]})
        result = TemporalFeatureTransformer().transform(df)
        assert "is_weekend" in result.columns
        assert all(result["is_weekend"] == 1)

    def test_weekday_not_weekend(self) -> None:
        df = pd.DataFrame({"day_of_week": [0, 1, 2, 3, 4]})
        result = TemporalFeatureTransformer().transform(df)
        assert all(result["is_weekend"] == 0)


class TestBuildingEncoderTransformerExtended:
    def test_known_building_type_encoded(self) -> None:
        df = pd.DataFrame({"building_type": ["residential"]})
        result = BuildingEncoderTransformer().transform(df)
        assert "building_intensity" in result.columns
        assert "building_type" not in result.columns

    def test_unknown_building_type_fallback(self) -> None:
        df = pd.DataFrame({"building_type": ["spaceship"]})
        result = BuildingEncoderTransformer().transform(df)
        assert result["building_intensity"].iloc[0] == pytest.approx(2.0)

    def test_no_building_type_column_unchanged(self) -> None:
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = BuildingEncoderTransformer().transform(df)
        assert "building_intensity" not in result.columns

    @pytest.mark.parametrize("bt", ["residential", "commercial", "industrial"])
    def test_various_known_types(self, bt: str) -> None:
        df = pd.DataFrame({"building_type": [bt]})
        result = BuildingEncoderTransformer().transform(df)
        assert result["building_intensity"].iloc[0] > 0.0
