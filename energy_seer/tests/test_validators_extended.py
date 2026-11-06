"""Extended tests for energy_seer app/validators.py — additional edge cases."""

from __future__ import annotations

import pytest

from app.validators import clamp, validate_meter_id


class TestClampExtended:
    def test_lo_equals_hi_returns_lo(self) -> None:
        assert clamp(5.0, 3.0, 3.0) == 3.0

    def test_negative_range(self) -> None:
        assert clamp(-2.0, -10.0, -1.0) == -2.0

    def test_value_at_lo_boundary(self) -> None:
        assert clamp(0.0, 0.0, 10.0) == 0.0

    def test_value_at_hi_boundary(self) -> None:
        assert clamp(10.0, 0.0, 10.0) == 10.0

    @pytest.mark.parametrize(
        "v,lo,hi,expected",
        [
            (float("inf"), 0.0, 100.0, 100.0),
            (float("-inf"), 0.0, 100.0, 0.0),
        ],
    )
    def test_infinity_clamped(self, v: float, lo: float, hi: float, expected: float) -> None:
        assert clamp(v, lo, hi) == expected


class TestValidateMeterIdExtended:
    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_meter_id("   ")

    def test_exactly_64_chars_accepted(self) -> None:
        meter_id = "M" * 64
        assert validate_meter_id(meter_id) == meter_id

    def test_65_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="64"):
            validate_meter_id("M" * 65)

    def test_leading_trailing_whitespace_stripped(self) -> None:
        assert validate_meter_id("  METER-01  ") == "METER-01"

    @pytest.mark.parametrize("meter_id", ["A", "1", "X" * 64])
    def test_valid_meter_ids(self, meter_id: str) -> None:
        assert validate_meter_id(meter_id) == meter_id.strip()
