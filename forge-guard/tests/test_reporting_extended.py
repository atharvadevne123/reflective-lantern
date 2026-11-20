"""Extended tests for forge-guard app/reporting.py helpers."""

from __future__ import annotations

import pytest

from app.reporting import format_defect_rate_pct


class TestFormatDefectRatePct:
    def test_zero(self) -> None:
        assert format_defect_rate_pct(0.0) == "0.00%"

    def test_one(self) -> None:
        assert format_defect_rate_pct(1.0) == "100.00%"

    def test_half(self) -> None:
        assert format_defect_rate_pct(0.5) == "50.00%"

    def test_small_value(self) -> None:
        result = format_defect_rate_pct(0.1234)
        assert "12.34" in result
        assert result.endswith("%")

    def test_returns_string(self) -> None:
        assert isinstance(format_defect_rate_pct(0.75), str)

    @pytest.mark.parametrize("rate,expected", [
        (0.0, "0.00%"),
        (0.25, "25.00%"),
        (0.5, "50.00%"),
        (0.75, "75.00%"),
        (1.0, "100.00%"),
    ])
    def test_parametrized_rates(self, rate: float, expected: str) -> None:
        assert format_defect_rate_pct(rate) == expected
