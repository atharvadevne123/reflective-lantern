"""Tests for Ops-Vision time-series forecasting module."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.forecasting import (
    ExponentialSmoothingForecaster,
    ForecastPoint,
    IncidentRateBuffer,
    get_rate_buffer,
)


class TestIncidentRateBuffer:
    """Tests for the IncidentRateBuffer accumulator."""

    def test_initial_buffer_is_empty(self):
        """New buffer has no counts."""
        buf = IncidentRateBuffer()
        assert len(buf.counts) == 0

    def test_record_adds_entry(self):
        """record() appends one entry to the buffer."""
        buf = IncidentRateBuffer()
        buf.record(datetime.utcnow(), 5)
        assert len(buf.counts) == 1

    def test_old_entries_are_pruned(self):
        """Entries older than window_hours are removed."""
        buf = IncidentRateBuffer(window_hours=1)
        old_ts = datetime.utcnow() - timedelta(hours=2)
        buf.record(old_ts, 3)
        buf.record(datetime.utcnow(), 1)
        assert len(buf.counts) == 1

    def test_as_array_returns_ndarray(self):
        """as_array() returns a numpy ndarray."""
        buf = IncidentRateBuffer()
        for i in range(5):
            buf.record(datetime.utcnow() + timedelta(hours=i), i)
        arr = buf.as_array()
        assert isinstance(arr, np.ndarray)

    def test_as_array_empty_buffer(self):
        """as_array() returns empty array for empty buffer."""
        buf = IncidentRateBuffer()
        arr = buf.as_array()
        assert len(arr) == 0


class TestExponentialSmoothingForecaster:
    """Tests for ExponentialSmoothingForecaster."""

    def _make_series(self, n: int = 48, trend: float = 0.0, noise: float = 0.5) -> np.ndarray:
        """Generate a synthetic incident-rate time series."""
        rng = np.random.default_rng(0)
        base = np.linspace(3.0, 3.0 + trend * n, n)
        return np.clip(base + rng.normal(0, noise, n), 0, None).astype(np.float64)

    def test_fit_returns_self(self):
        """fit() returns the forecaster instance."""
        f = ExponentialSmoothingForecaster()
        result = f.fit(self._make_series())
        assert result is f

    def test_forecast_length_equals_horizon(self):
        """forecast() returns exactly horizon points."""
        f = ExponentialSmoothingForecaster(horizon=12)
        f.fit(self._make_series())
        points = f.forecast(datetime.utcnow())
        assert len(points) == 12

    def test_forecast_returns_forecast_points(self):
        """Each forecast element is a ForecastPoint."""
        f = ExponentialSmoothingForecaster(horizon=5)
        f.fit(self._make_series())
        points = f.forecast(datetime.utcnow())
        for p in points:
            assert isinstance(p, ForecastPoint)

    def test_forecast_timestamps_are_ordered(self):
        """Forecast timestamps must be strictly increasing."""
        f = ExponentialSmoothingForecaster(horizon=10)
        f.fit(self._make_series())
        base = datetime.utcnow()
        points = f.forecast(base)
        for i in range(1, len(points)):
            assert points[i].timestamp > points[i - 1].timestamp

    def test_lower_bound_le_value_le_upper_bound(self):
        """Each ForecastPoint must satisfy lower <= value <= upper."""
        f = ExponentialSmoothingForecaster(horizon=24)
        f.fit(self._make_series(trend=0.1))
        for p in f.forecast(datetime.utcnow()):
            assert p.lower_bound <= p.value <= p.upper_bound

    def test_values_are_non_negative(self):
        """Forecast values (incident rates) must be >= 0."""
        f = ExponentialSmoothingForecaster(horizon=24)
        f.fit(self._make_series())
        for p in f.forecast(datetime.utcnow()):
            assert p.value >= 0.0

    def test_forecast_raises_if_not_fitted(self):
        """forecast() raises RuntimeError if fit() was not called."""
        f = ExponentialSmoothingForecaster()
        with pytest.raises(RuntimeError, match="Forecaster must be fitted"):
            f.forecast(datetime.utcnow())

    def test_short_series_handled(self):
        """fit() handles a 1-element series without error."""
        f = ExponentialSmoothingForecaster(horizon=3)
        f.fit(np.array([5.0]))
        points = f.forecast(datetime.utcnow())
        assert len(points) == 3

    @pytest.mark.parametrize("horizon", [6, 12, 24])
    def test_various_horizons(self, horizon):
        """Forecaster works for different horizon values."""
        f = ExponentialSmoothingForecaster(horizon=horizon)
        f.fit(self._make_series(n=50))
        points = f.forecast(datetime.utcnow())
        assert len(points) == horizon


class TestExponentialSmoothingForecasterProperties:
    """Tests for is_fitted property and params dict."""

    def test_is_fitted_false_before_fit(self):
        """is_fitted is False before calling fit()."""
        f = ExponentialSmoothingForecaster()
        assert f.is_fitted is False

    def test_is_fitted_true_after_fit(self):
        """is_fitted becomes True after successful fit()."""
        f = ExponentialSmoothingForecaster()
        f.fit(np.array([1.0, 2.0, 3.0]))
        assert f.is_fitted is True

    def test_params_contains_alpha(self):
        """params dict has alpha key matching constructor arg."""
        f = ExponentialSmoothingForecaster(alpha=0.5)
        assert f.params["alpha"] == 0.5

    def test_params_contains_beta(self):
        """params dict has beta key matching constructor arg."""
        f = ExponentialSmoothingForecaster(beta=0.2)
        assert f.params["beta"] == 0.2

    def test_params_contains_horizon(self):
        """params dict has horizon key matching constructor arg."""
        f = ExponentialSmoothingForecaster(horizon=12)
        assert f.params["horizon"] == 12


class TestIncidentRateBufferResetAndLen:
    """Tests for IncidentRateBuffer.reset() and __len__()."""

    def test_reset_clears_all_counts(self):
        """reset() removes all entries from the buffer."""
        buf = IncidentRateBuffer()
        for i in range(5):
            buf.record(datetime.utcnow() + timedelta(hours=i), i + 1)
        buf.reset()
        assert len(buf.counts) == 0

    def test_reset_allows_new_records(self):
        """Buffer is usable after reset."""
        buf = IncidentRateBuffer()
        buf.record(datetime.utcnow(), 3)
        buf.reset()
        buf.record(datetime.utcnow(), 7)
        assert len(buf) == 1

    def test_len_reflects_count(self):
        """__len__ returns number of entries in the buffer."""
        buf = IncidentRateBuffer()
        for i in range(4):
            buf.record(datetime.utcnow() + timedelta(hours=i), 1)
        assert len(buf) == 4

    def test_len_zero_on_empty_buffer(self):
        """__len__ returns 0 for a new buffer."""
        buf = IncidentRateBuffer()
        assert len(buf) == 0

    def test_len_decreases_after_reset(self):
        """__len__ returns 0 after reset."""
        buf = IncidentRateBuffer()
        buf.record(datetime.utcnow(), 2)
        buf.reset()
        assert len(buf) == 0


class TestGetRateBufferSingleton:
    """Tests for the get_rate_buffer() singleton helper."""

    def test_returns_incident_rate_buffer(self):
        """get_rate_buffer() returns an IncidentRateBuffer."""
        buf = get_rate_buffer()
        assert isinstance(buf, IncidentRateBuffer)

    def test_returns_same_instance(self):
        """get_rate_buffer() returns the same object on repeated calls."""
        b1 = get_rate_buffer()
        b2 = get_rate_buffer()
        assert b1 is b2
