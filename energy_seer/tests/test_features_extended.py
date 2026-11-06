"""Extended tests for energy_seer app/features.py."""

from __future__ import annotations

import pandas as pd
import pytest

from app.features import (
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
