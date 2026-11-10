"""Extended tests for ops-vision app/features.py transformers and helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    clip_to_unit_interval,
    dataframe_from_dict,
    validate_positive_float,
)


class TestValidatePositiveFloat:
    def test_valid_value_returned(self) -> None:
        assert validate_positive_float(1.5, "latency") == 1.5

    def test_zero_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            validate_positive_float(0.0, "latency")

    def test_negative_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            validate_positive_float(-0.1, "cpu")

    def test_large_value_ok(self) -> None:
        assert validate_positive_float(1_000_000.0, "bytes") == 1_000_000.0


class TestClipToUnitInterval:
    def test_value_in_range_unchanged(self) -> None:
        assert clip_to_unit_interval(0.5) == pytest.approx(0.5)

    def test_below_zero_clips_to_zero(self) -> None:
        assert clip_to_unit_interval(-1.0) == pytest.approx(0.0)

    def test_above_one_clips_to_one(self) -> None:
        assert clip_to_unit_interval(2.0) == pytest.approx(1.0)

    def test_zero_boundary(self) -> None:
        assert clip_to_unit_interval(0.0) == pytest.approx(0.0)

    def test_one_boundary(self) -> None:
        assert clip_to_unit_interval(1.0) == pytest.approx(1.0)


class TestDataframeFromDict:
    def test_returns_dataframe(self) -> None:
        payload = {"cpu_util": 0.5, "mem_util": 0.4, "latency_ms": 100.0}
        df = dataframe_from_dict(payload)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        payload = {"cpu_util": 0.5, "mem_util": 0.4}
        df = dataframe_from_dict(payload)
        assert "cpu_util" in df.columns
        assert "mem_util" in df.columns

    def test_has_one_row(self) -> None:
        payload = {"cpu_util": 0.5}
        df = dataframe_from_dict(payload)
        assert len(df) == 1

    def test_values_match_payload(self) -> None:
        payload = {"cpu_util": 0.75, "latency_ms": 200.0}
        df = dataframe_from_dict(payload)
        assert df["cpu_util"].iloc[0] == pytest.approx(0.75)
