"""Tests for Ops-Vision time-series forecasting module."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.forecasting import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_HORIZON,
    DEFAULT_STEP_HOURS,
    ExponentialSmoothingForecaster,
    ForecastPoint,
    IncidentRateBuffer,
    forecast_to_dicts,
    get_rate_buffer,
)


class TestForecastToDicts:
    """Tests for forecast_to_dicts() batch helper."""

    def _make_point(self, value: float = 1.0) -> ForecastPoint:
        return ForecastPoint(
            timestamp=datetime.utcnow(),
            value=value,
            lower_bound=0.0,
            upper_bound=value + 1.0,
        )

    def test_empty_list_returns_empty(self) -> None:
        """forecast_to_dicts([]) returns []."""
        assert forecast_to_dicts([]) == []

    def test_single_point_returns_one_dict(self) -> None:
        """forecast_to_dicts() with one point returns a list of one dict."""
        result = forecast_to_dicts([self._make_point()])
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_multiple_points_correct_length(self) -> None:
        """forecast_to_dicts() length matches input list length."""
        points = [self._make_point(float(i)) for i in range(5)]
        result = forecast_to_dicts(points)
        assert len(result) == 5

    @pytest.mark.parametrize("n", [1, 3, 6, 12, 24])
    def test_length_parametrized(self, n: int) -> None:
        """forecast_to_dicts always returns exactly n dicts for n input points."""
        points = [self._make_point() for _ in range(n)]
        assert len(forecast_to_dicts(points)) == n


class TestForecastPointToDict:
    """Tests for ForecastPoint.to_dict()."""

    def test_to_dict_returns_dict(self) -> None:
        """to_dict() returns a dict."""
        fp = ForecastPoint(
            timestamp=datetime.utcnow(),
            value=3.0,
            lower_bound=1.0,
            upper_bound=5.0,
        )
        assert isinstance(fp.to_dict(), dict)

    def test_to_dict_has_timestamp_key(self) -> None:
        """to_dict() includes 'timestamp' key."""
        fp = ForecastPoint(
            timestamp=datetime.utcnow(),
            value=1.0,
            lower_bound=0.0,
            upper_bound=2.0,
        )
        assert "timestamp" in fp.to_dict()

    def test_to_dict_timestamp_is_iso_string(self) -> None:
        """to_dict() timestamp is an ISO-format string."""
        fp = ForecastPoint(
            timestamp=datetime(2026, 1, 15, 12, 0, 0),
            value=2.0,
            lower_bound=1.0,
            upper_bound=3.0,
        )
        d = fp.to_dict()
        assert isinstance(d["timestamp"], str)
        assert "2026" in d["timestamp"]

    @pytest.mark.parametrize("value", [0.0, 1.5, 3.0, 10.0])
    def test_to_dict_value_matches(self, value: float) -> None:
        """to_dict() value key matches the ForecastPoint value."""
        fp = ForecastPoint(
            timestamp=datetime.utcnow(),
            value=value,
            lower_bound=0.0,
            upper_bound=value + 1.0,
        )
        assert fp.to_dict()["value"] == value

    def test_to_dict_has_all_keys(self) -> None:
        """to_dict() output has timestamp, value, lower_bound, upper_bound."""
        fp = ForecastPoint(
            timestamp=datetime.utcnow(),
            value=1.0,
            lower_bound=0.0,
            upper_bound=2.0,
        )
        assert set(fp.to_dict()) == {"timestamp", "value", "lower_bound", "upper_bound"}


class TestForecastingConstants:
    """Tests for module-level constants."""

    def test_default_alpha_in_range(self) -> None:
        """DEFAULT_ALPHA is strictly between 0 and 1."""
        assert 0.0 < DEFAULT_ALPHA < 1.0

    def test_default_beta_in_range(self) -> None:
        """DEFAULT_BETA is strictly between 0 and 1."""
        assert 0.0 < DEFAULT_BETA < 1.0

    def test_default_horizon_positive(self) -> None:
        """DEFAULT_HORIZON is a positive integer."""
        assert DEFAULT_HORIZON > 0

    def test_default_step_hours_positive(self) -> None:
        """DEFAULT_STEP_HOURS is a positive integer."""
        assert DEFAULT_STEP_HOURS > 0

    def test_forecaster_uses_default_alpha(self) -> None:
        """ExponentialSmoothingForecaster default alpha matches the constant."""
        f = ExponentialSmoothingForecaster()
        assert f.alpha == DEFAULT_ALPHA

    def test_forecaster_uses_default_horizon(self) -> None:
        """ExponentialSmoothingForecaster default horizon matches the constant."""
        f = ExponentialSmoothingForecaster()
        assert f.horizon == DEFAULT_HORIZON


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


class TestIncidentRateBufferGetCountsSince:
    """Tests for IncidentRateBuffer.get_counts_since()."""

    def test_returns_list(self) -> None:
        """get_counts_since() returns a list."""
        buf = IncidentRateBuffer()
        buf.record(datetime.utcnow(), 3)
        result = buf.get_counts_since(datetime.utcnow() - timedelta(hours=1))
        assert isinstance(result, list)

    def test_filters_old_entries(self) -> None:
        """get_counts_since() excludes entries before the cutoff."""
        buf = IncidentRateBuffer()
        now = datetime.utcnow()
        buf.record(now - timedelta(hours=5), 10)
        buf.record(now, 3)
        result = buf.get_counts_since(now - timedelta(hours=1))
        assert len(result) == 1
        assert result[0][1] == 3

    def test_returns_sorted_by_timestamp(self) -> None:
        """get_counts_since() returns entries sorted by timestamp."""
        buf = IncidentRateBuffer()
        now = datetime.utcnow()
        buf.record(now + timedelta(hours=2), 5)
        buf.record(now, 1)
        buf.record(now + timedelta(hours=1), 3)
        result = buf.get_counts_since(now - timedelta(hours=1))
        timestamps = [t for t, _ in result]
        assert timestamps == sorted(timestamps)

    def test_empty_buffer_returns_empty(self) -> None:
        """get_counts_since() on empty buffer returns []."""
        buf = IncidentRateBuffer()
        assert buf.get_counts_since(datetime.utcnow()) == []

    @pytest.mark.parametrize("hours_back", [1, 6, 12, 24])
    def test_cutoff_hours_parametrized(self, hours_back: int) -> None:
        """get_counts_since() with varying cutoffs returns only recent entries."""
        buf = IncidentRateBuffer()
        now = datetime.utcnow()
        buf.record(now - timedelta(hours=hours_back + 1), 100)
        buf.record(now - timedelta(hours=1), 5)
        cutoff = now - timedelta(hours=hours_back)
        result = buf.get_counts_since(cutoff)
        assert all(t >= cutoff for t, _ in result)


class TestIncidentRateBufferRepr:
    """Tests for IncidentRateBuffer.__repr__."""

    def test_repr_is_string(self) -> None:
        """repr() returns a str."""
        buf = IncidentRateBuffer()
        assert isinstance(repr(buf), str)

    def test_repr_contains_len(self) -> None:
        """repr includes the current buffer length."""
        buf = IncidentRateBuffer()
        buf.record(datetime.utcnow(), 3)
        assert "1" in repr(buf)

    def test_repr_contains_window_hours(self) -> None:
        """repr includes the window_hours configuration."""
        buf = IncidentRateBuffer(window_hours=48)
        assert "48" in repr(buf)

    @pytest.mark.parametrize("window_hours", [24, 48, 72, 168])
    def test_repr_window_hours_parametrized(self, window_hours: int) -> None:
        """repr contains the window_hours for any configured value."""
        buf = IncidentRateBuffer(window_hours=window_hours)
        assert str(window_hours) in repr(buf)


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
        assert all(t1 < t2 for t1, t2 in zip(timestamps[:-1], timestamps[1:], strict=False))

    @pytest.mark.parametrize("incident_count", [0, 1, 5, 10])
    def test_buffer_records_correct_count(self, incident_count: int) -> None:
        """IncidentRateBuffer records the right sum across intervals."""
        buf = IncidentRateBuffer()
        now = datetime.utcnow()
        buf.record(now, incident_count)
        arr = buf.as_array()
        assert arr.sum() == incident_count


class TestIncidentRateBufferLastTimestamp:
    """Tests for IncidentRateBuffer.last_timestamp property."""

    def test_last_timestamp_none_on_empty_buffer(self) -> None:
        """last_timestamp is None when the buffer is empty."""
        buf = IncidentRateBuffer()
        assert buf.last_timestamp is None

    def test_last_timestamp_returns_datetime(self) -> None:
        """last_timestamp returns a datetime after a record is added."""
        from datetime import datetime
        buf = IncidentRateBuffer()
        ts = datetime(2024, 1, 1, 12, 0)
        buf.record(ts, 3)
        assert isinstance(buf.last_timestamp, datetime)

    def test_last_timestamp_is_max_of_recorded_timestamps(self) -> None:
        """last_timestamp equals the latest recorded timestamp."""
        from datetime import datetime, timedelta
        buf = IncidentRateBuffer()
        t0 = datetime(2024, 1, 1, 10, 0)
        t1 = t0 + timedelta(hours=1)
        t2 = t0 + timedelta(hours=2)
        buf.record(t0, 1)
        buf.record(t2, 3)
        buf.record(t1, 2)
        assert buf.last_timestamp == t2

    @pytest.mark.parametrize("hours_offset", [1, 5, 12, 24])
    def test_last_timestamp_advances_parametrized(self, hours_offset: int) -> None:
        """last_timestamp moves forward when a later record is added."""
        from datetime import datetime, timedelta
        buf = IncidentRateBuffer()
        t0 = datetime(2024, 6, 1, 0, 0)
        t1 = t0 + timedelta(hours=hours_offset)
        buf.record(t0, 1)
        buf.record(t1, 2)
        assert buf.last_timestamp == t1
