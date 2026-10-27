"""Extended tests for ops-vision app/forecasting.py."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.forecasting import ForecastPoint, IncidentRateBuffer


class TestForecastPoint:
    def test_create_instance(self) -> None:
        fp = ForecastPoint(
            timestamp=datetime(2024, 1, 1, 12, 0),
            value=3.5,
            lower_bound=1.0,
            upper_bound=6.0,
        )
        assert fp.value == pytest.approx(3.5)

    def test_lower_lt_upper(self) -> None:
        fp = ForecastPoint(
            timestamp=datetime(2024, 1, 1),
            value=2.0,
            lower_bound=1.0,
            upper_bound=3.0,
        )
        assert fp.lower_bound < fp.upper_bound


class TestIncidentRateBuffer:
    def test_initial_len_zero(self) -> None:
        buf = IncidentRateBuffer()
        assert len(buf) == 0

    def test_record_increments_len(self) -> None:
        buf = IncidentRateBuffer()
        buf.record(datetime(2024, 1, 1, 10, 0), 5)
        assert len(buf) == 1

    def test_as_array_empty(self) -> None:
        buf = IncidentRateBuffer()
        arr = buf.as_array()
        assert arr.dtype == np.float64
        assert len(arr) == 0

    def test_as_array_values(self) -> None:
        buf = IncidentRateBuffer()
        now = datetime(2024, 1, 1, 10, 0)
        buf.record(now, 3)
        buf.record(now + timedelta(hours=1), 5)
        arr = buf.as_array()
        assert len(arr) == 2
        assert 3.0 in arr
        assert 5.0 in arr

    def test_prunes_old_entries(self) -> None:
        buf = IncidentRateBuffer(window_hours=2)
        now = datetime(2024, 1, 1, 10, 0)
        buf.record(now - timedelta(hours=3), 10)
        buf.record(now, 5)
        assert len(buf) == 1

    def test_reset_clears_buffer(self) -> None:
        buf = IncidentRateBuffer()
        buf.record(datetime(2024, 1, 1), 5)
        buf.reset()
        assert len(buf) == 0

    def test_total_incidents_sum(self) -> None:
        buf = IncidentRateBuffer()
        now = datetime(2024, 1, 1, 10, 0)
        buf.record(now, 3)
        buf.record(now + timedelta(hours=1), 7)
        assert buf.total_incidents == 10

    def test_total_incidents_empty(self) -> None:
        buf = IncidentRateBuffer()
        assert buf.total_incidents == 0
