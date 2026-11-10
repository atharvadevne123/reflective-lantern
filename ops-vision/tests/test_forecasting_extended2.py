"""Extended tests for ops-vision app/forecasting.py IncidentRateBuffer."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.forecasting import IncidentRateBuffer


def _ts(offset_hours: int = 0) -> datetime:
    base = datetime(2026, 1, 1, 12, 0, 0)
    return base + timedelta(hours=offset_hours)


class TestIncidentRateBufferRecord:
    def test_len_increases(self) -> None:
        buf = IncidentRateBuffer()
        buf.record(_ts(0), 5)
        assert len(buf) == 1

    def test_prunes_old_entries(self) -> None:
        buf = IncidentRateBuffer()
        buf.window_hours = 2
        buf.record(_ts(0), 3)
        buf.record(_ts(3), 7)
        assert len(buf) == 1

    def test_keeps_within_window(self) -> None:
        buf = IncidentRateBuffer()
        buf.window_hours = 10
        for i in range(5):
            buf.record(_ts(i), i)
        assert len(buf) == 5


class TestIncidentRateBufferAsArray:
    def test_empty_returns_empty_array(self) -> None:
        buf = IncidentRateBuffer()
        arr = buf.as_array()
        assert len(arr) == 0

    def test_values_match_records(self) -> None:
        buf = IncidentRateBuffer()
        for i, count in enumerate([3, 5, 2]):
            buf.record(_ts(i), count)
        arr = buf.as_array()
        assert sorted(arr.tolist()) == sorted([3.0, 5.0, 2.0])


class TestIncidentRateBufferProperties:
    def test_total_incidents(self) -> None:
        buf = IncidentRateBuffer()
        buf.record(_ts(0), 3)
        buf.record(_ts(1), 7)
        assert buf.total_incidents == 10

    def test_last_timestamp(self) -> None:
        buf = IncidentRateBuffer()
        t0 = _ts(0)
        t1 = _ts(1)
        buf.record(t0, 1)
        buf.record(t1, 2)
        assert buf.last_timestamp == t1

    def test_last_timestamp_none_on_empty(self) -> None:
        buf = IncidentRateBuffer()
        assert buf.last_timestamp is None

    def test_window_span_zero_on_one_entry(self) -> None:
        buf = IncidentRateBuffer()
        buf.record(_ts(0), 1)
        assert buf.window_span_hours == pytest.approx(0.0)

    def test_window_span_on_two_entries(self) -> None:
        buf = IncidentRateBuffer()
        buf.record(_ts(0), 1)
        buf.record(_ts(2), 1)
        assert buf.window_span_hours == pytest.approx(2.0)


class TestIncidentRateBufferReset:
    def test_reset_clears_all(self) -> None:
        buf = IncidentRateBuffer()
        buf.record(_ts(0), 5)
        buf.reset()
        assert len(buf) == 0

    def test_reset_total_incidents_zero(self) -> None:
        buf = IncidentRateBuffer()
        buf.record(_ts(0), 10)
        buf.reset()
        assert buf.total_incidents == 0
