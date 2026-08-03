"""Tests for validators, cache, telemetry, forecaster, demand spike."""
from __future__ import annotations

import pytest


class TestValidators:
    def test_clamp_within_bounds(self):
        from app.validators import clamp
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamp_below_lo(self):
        from app.validators import clamp
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_clamp_above_hi(self):
        from app.validators import clamp
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_validate_meter_id_valid(self):
        from app.validators import validate_meter_id
        assert validate_meter_id("  meter_001  ") == "meter_001"

    def test_validate_meter_id_empty_raises(self):
        from app.validators import validate_meter_id
        with pytest.raises(ValueError, match="empty"):
            validate_meter_id("   ")

    def test_validate_feature_window_too_short(self):
        from app.validators import validate_feature_window
        with pytest.raises(ValueError):
            validate_feature_window([1.0, 2.0])


class TestCache:
    def test_set_and_get(self):
        from app.cache import get_cache, set_cache
        set_cache("k1", 42, ttl=10.0)
        assert get_cache("k1") == 42

    def test_miss_returns_none(self):
        from app.cache import get_cache
        assert get_cache("nonexistent_key_xyz") is None

    def test_invalidate(self):
        from app.cache import get_cache, invalidate, set_cache
        set_cache("k2", "hello", ttl=10.0)
        invalidate("k2")
        assert get_cache("k2") is None

    def test_cache_size(self):
        from app.cache import cache_size, clear_all, set_cache
        clear_all()
        set_cache("a", 1, ttl=10.0)
        set_cache("b", 2, ttl=10.0)
        assert cache_size() == 2


class TestTelemetry:
    def test_increment_and_get(self):
        from app.telemetry import get, increment, reset
        reset("req_count")
        increment("req_count", 3)
        assert get("req_count") == 3

    def test_snapshot(self):
        from app.telemetry import increment, reset, snapshot
        reset()
        increment("x", 5)
        s = snapshot()
        assert s["x"] == 5


class TestDemandSpike:
    def test_detects_spike(self):
        from app.demand_spike import detect_spike
        vals = [3.0] * 20 + [50.0]
        result = detect_spike(vals)
        assert result["spike_count"] > 0
        assert 20 in result["spike_indices"]

    def test_no_spike_flat(self):
        from app.demand_spike import detect_spike
        vals = [3.0] * 30
        result = detect_spike(vals)
        assert result["spike_count"] == 0

    def test_insufficient_data(self):
        from app.demand_spike import detect_spike
        result = detect_spike([1.0, 2.0])
        assert result["method"] == "insufficient_data"

    @pytest.mark.parametrize("n", [10, 20, 50])
    def test_various_lengths(self, n):
        from app.demand_spike import detect_spike
        result = detect_spike(list(range(n)))
        assert "spike_count" in result


class TestForecaster:
    def test_linear_trend(self):
        from app.forecaster import linear_trend
        result = linear_trend([1.0, 2.0, 3.0, 4.0])
        assert result["slope"] > 0
        assert result["next"] > 4.0

    def test_moving_average_forecast_length(self):
        from app.forecaster import moving_average_forecast
        preds = moving_average_forecast(list(range(24)), steps=6)
        assert len(preds) == 6

    def test_seasonal_decompose_returns_keys(self):
        from app.forecaster import seasonal_decompose_simple
        vals = list(range(48))
        result = seasonal_decompose_simple(vals, period=24)
        assert "seasonal" in result
        assert "trend" in result

    def test_forecast_positive_for_positive_input(self):
        from app.forecaster import moving_average_forecast
        preds = moving_average_forecast([5.0] * 24, steps=3)
        assert all(p > 0 for p in preds)
