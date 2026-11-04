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

    def test_initial_buffer_is_empty(self) -> None:
        """New buffer has no counts."""
        buf = IncidentRateBuffer()
        assert len(buf.counts) == 0

    def test_record_adds_entry(self) -> None:
        """record() appends one entry to the buffer."""
        buf = IncidentRateBuffer()
        buf.record(datetime.utcnow(), 5)
        assert len(buf.counts) == 1

    def test_old_entries_are_pruned(self) -> None:
        """Entries older than window_hours are removed."""
        buf = IncidentRateBuffer(window_hours=1)
        old_ts = datetime.utcnow() - timedelta(hours=2)
        buf.record(old_ts, 3)
        buf.record(datetime.utcnow(), 1)
        assert len(buf.counts) == 1

    def test_as_array_returns_ndarray(self) -> None:
        """as_array() returns a numpy ndarray."""
        buf = IncidentRateBuffer()
        for i in range(5):
            buf.record(datetime.utcnow() + timedelta(hours=i), i)
        arr = buf.as_array()
        assert isinstance(arr, np.ndarray)

    def test_as_array_empty_buffer(self) -> None:
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

    def test_fit_returns_self(self) -> None:
        """fit() returns the forecaster instance."""
        f = ExponentialSmoothingForecaster()
        result = f.fit(self._make_series())
        assert result is f

    def test_forecast_length_equals_horizon(self) -> None:
        """forecast() returns exactly horizon points."""
        f = ExponentialSmoothingForecaster(horizon=12)
        f.fit(self._make_series())
        points = f.forecast(datetime.utcnow())
        assert len(points) == 12

    def test_forecast_returns_forecast_points(self) -> None:
        """Each forecast element is a ForecastPoint."""
        f = ExponentialSmoothingForecaster(horizon=5)
        f.fit(self._make_series())
        points = f.forecast(datetime.utcnow())
        for p in points:
            assert isinstance(p, ForecastPoint)

    def test_forecast_timestamps_are_ordered(self) -> None:
        """Forecast timestamps must be strictly increasing."""
        f = ExponentialSmoothingForecaster(horizon=10)
        f.fit(self._make_series())
        base = datetime.utcnow()
        points = f.forecast(base)
        for i in range(1, len(points)):
            assert points[i].timestamp > points[i - 1].timestamp

    def test_lower_bound_le_value_le_upper_bound(self) -> None:
        """Each ForecastPoint must satisfy lower <= value <= upper."""
        f = ExponentialSmoothingForecaster(horizon=24)
        f.fit(self._make_series(trend=0.1))
        for p in f.forecast(datetime.utcnow()):
            assert p.lower_bound <= p.value <= p.upper_bound

    def test_values_are_non_negative(self) -> None:
        """Forecast values (incident rates) must be >= 0."""
        f = ExponentialSmoothingForecaster(horizon=24)
        f.fit(self._make_series())
        for p in f.forecast(datetime.utcnow()):
            assert p.value >= 0.0

    def test_forecast_raises_if_not_fitted(self) -> None:
        """forecast() raises RuntimeError if fit() was not called."""
        f = ExponentialSmoothingForecaster()
        with pytest.raises(RuntimeError, match="Forecaster must be fitted"):
            f.forecast(datetime.utcnow())

    def test_short_series_handled(self) -> None:
        """fit() handles a 1-element series without error."""
        f = ExponentialSmoothingForecaster(horizon=3)
        f.fit(np.array([5.0]))
        points = f.forecast(datetime.utcnow())
        assert len(points) == 3

    @pytest.mark.parametrize("horizon", [6, 12, 24])
    def test_various_horizons(self, horizon) -> None:
        """Forecaster works for different horizon values."""
        f = ExponentialSmoothingForecaster(horizon=horizon)
        f.fit(self._make_series(n=50))
        points = f.forecast(datetime.utcnow())
        assert len(points) == horizon


class TestIncidentRateBufferProperties:
    """Tests for total_incidents and window_span_hours properties."""

    def test_total_incidents_sums_counts(self) -> None:
        """total_incidents sums all counts in the buffer."""
        buf = IncidentRateBuffer()
        buf.record(datetime.utcnow(), 3)
        buf.record(datetime.utcnow() + timedelta(hours=1), 7)
        assert buf.total_incidents == 10

    def test_total_incidents_zero_on_empty(self) -> None:
        """total_incidents is 0 for an empty buffer."""
        buf = IncidentRateBuffer()
        assert buf.total_incidents == 0

    def test_window_span_zero_for_one_entry(self) -> None:
        """window_span_hours is 0.0 when fewer than 2 entries exist."""
        buf = IncidentRateBuffer()
        buf.record(datetime.utcnow(), 1)
        assert buf.window_span_hours == 0.0

    def test_window_span_correct_for_multiple_entries(self) -> None:
        """window_span_hours reflects the spread between oldest and newest entries."""
        buf = IncidentRateBuffer()
        base = datetime(2026, 8, 31, 0, 0)
        buf.record(base, 1)
        buf.record(base + timedelta(hours=5), 2)
        assert abs(buf.window_span_hours - 5.0) < 0.01


class TestExponentialSmoothingForecasterProperties:
    """Tests for is_fitted property and params dict."""

    def test_is_fitted_false_before_fit(self) -> None:
        """is_fitted is False before calling fit()."""
        f = ExponentialSmoothingForecaster()
        assert f.is_fitted is False

    def test_is_fitted_true_after_fit(self) -> None:
        """is_fitted becomes True after successful fit()."""
        f = ExponentialSmoothingForecaster()
        f.fit(np.array([1.0, 2.0, 3.0]))
        assert f.is_fitted is True

    def test_params_contains_alpha(self) -> None:
        """params dict has alpha key matching constructor arg."""
        f = ExponentialSmoothingForecaster(alpha=0.5)
        assert f.params["alpha"] == 0.5

    def test_params_contains_beta(self) -> None:
        """params dict has beta key matching constructor arg."""
        f = ExponentialSmoothingForecaster(beta=0.2)
        assert f.params["beta"] == 0.2

    def test_params_contains_horizon(self) -> None:
        """params dict has horizon key matching constructor arg."""
        f = ExponentialSmoothingForecaster(horizon=12)
        assert f.params["horizon"] == 12


class TestIncidentRateBufferResetAndLen:
    """Tests for IncidentRateBuffer.reset() and __len__()."""

    def test_reset_clears_all_counts(self) -> None:
        """reset() removes all entries from the buffer."""
        buf = IncidentRateBuffer()
        for i in range(5):
            buf.record(datetime.utcnow() + timedelta(hours=i), i + 1)
        buf.reset()
        assert len(buf.counts) == 0

    def test_reset_allows_new_records(self) -> None:
        """Buffer is usable after reset."""
        buf = IncidentRateBuffer()
        buf.record(datetime.utcnow(), 3)
        buf.reset()
        buf.record(datetime.utcnow(), 7)
        assert len(buf) == 1

    def test_len_reflects_count(self) -> None:
        """__len__ returns number of entries in the buffer."""
        buf = IncidentRateBuffer()
        for i in range(4):
            buf.record(datetime.utcnow() + timedelta(hours=i), 1)
        assert len(buf) == 4

    def test_len_zero_on_empty_buffer(self) -> None:
        """__len__ returns 0 for a new buffer."""
        buf = IncidentRateBuffer()
        assert len(buf) == 0

    def test_len_decreases_after_reset(self) -> None:
        """__len__ returns 0 after reset."""
        buf = IncidentRateBuffer()
        buf.record(datetime.utcnow(), 2)
        buf.reset()
        assert len(buf) == 0


class TestExponentialSmoothingForecasterRepr:
    """Tests for ExponentialSmoothingForecaster.__repr__."""

    def test_repr_is_string(self) -> None:
        """repr() returns a string."""
        assert isinstance(repr(ExponentialSmoothingForecaster()), str)

    def test_repr_contains_alpha(self) -> None:
        """repr includes the alpha parameter."""
        f = ExponentialSmoothingForecaster(alpha=0.4)
        assert "0.4" in repr(f)

    def test_repr_contains_horizon(self) -> None:
        """repr includes the horizon parameter."""
        f = ExponentialSmoothingForecaster(horizon=18)
        assert "18" in repr(f)

    def test_repr_shows_fitted_state(self) -> None:
        """repr shows 'fitted' after calling fit()."""
        f = ExponentialSmoothingForecaster()
        f.fit(np.array([1.0, 2.0, 3.0]))
        assert "fitted" in repr(f)

    def test_repr_shows_unfitted_before_fit(self) -> None:
        """repr shows 'unfitted' before calling fit()."""
        f = ExponentialSmoothingForecaster()
        assert "unfitted" in repr(f)

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.7, 0.9])
    def test_repr_alpha_parametrized(self, alpha: float) -> None:
        """repr contains the alpha value for any configured alpha."""
        f = ExponentialSmoothingForecaster(alpha=alpha)
        assert str(alpha) in repr(f)


class TestGetRateBufferSingleton:
    """Tests for the get_rate_buffer() singleton helper."""

    def test_returns_incident_rate_buffer(self) -> None:
        """get_rate_buffer() returns an IncidentRateBuffer."""
        buf = get_rate_buffer()
        assert isinstance(buf, IncidentRateBuffer)

    def test_returns_same_instance(self) -> None:
        """get_rate_buffer() returns the same object on repeated calls."""
        b1 = get_rate_buffer()
        b2 = get_rate_buffer()
        assert b1 is b2


class TestExponentialSmoothingParametrized:
    """Parametrized tests for ExponentialSmoothingForecaster."""

    @pytest.mark.parametrize("horizon", [6, 12, 24, 48])
    def test_forecast_length_equals_horizon(self, horizon: int) -> None:
        """forecast() returns exactly horizon ForecastPoint objects."""
        forecaster = ExponentialSmoothingForecaster(horizon=horizon)
        series = np.array([1.0, 2.0, 1.5, 3.0, 2.0, 1.0, 2.5], dtype=float)
        forecaster.fit(series)
        points = forecaster.forecast(base_time=datetime.utcnow())
        assert len(points) == horizon

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9])
    def test_forecast_with_various_alpha(self, alpha: float) -> None:
        """ExponentialSmoothingForecaster accepts different alpha values."""
        forecaster = ExponentialSmoothingForecaster(horizon=6, alpha=alpha)
        series = np.array([2.0, 3.0, 2.5, 4.0, 3.0, 2.0], dtype=float)
        forecaster.fit(series)
        points = forecaster.forecast(base_time=datetime.utcnow())
        assert all(isinstance(p, ForecastPoint) for p in points)

    @pytest.mark.parametrize("n_points", [2, 5, 10, 20])
    def test_forecast_timestamps_are_monotone(self, n_points: int) -> None:
        """forecast() timestamps are strictly increasing."""
        forecaster = ExponentialSmoothingForecaster(horizon=n_points)
        series = np.ones(10, dtype=float)
        forecaster.fit(series)
        points = forecaster.forecast(base_time=datetime.utcnow())
        timestamps = [p.timestamp for p in points]
        assert all(t1 < t2 for t1, t2 in zip(timestamps[:-1], timestamps[1:]))

    @pytest.mark.parametrize("incident_count", [0, 1, 5, 10])
    def test_buffer_records_correct_count(self, incident_count: int) -> None:
        """IncidentRateBuffer records the right sum across intervals."""
        buf = IncidentRateBuffer()
        now = datetime.utcnow()
        buf.record(now, incident_count)
        arr = buf.as_array()
        assert arr.sum() == incident_count
