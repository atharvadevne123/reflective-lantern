"""Extended forecasting tests with parametrize and boundary coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.forecasting import (
    ExponentialSmoothingForecaster,
    ForecastPoint,
    IncidentRateBuffer,
    forecast_to_dicts,
)


def _make_buffer_with_data(n: int = 24, base_count: int = 5) -> IncidentRateBuffer:
    buf = IncidentRateBuffer()
    base_ts = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(n):
        buf.record(base_ts + timedelta(hours=i), base_count)
    return buf


def test_incident_rate_buffer_record_stores_data() -> None:
    buf = IncidentRateBuffer()
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    buf.record(ts, 3)
    assert len(buf.counts) == 1


def test_incident_rate_buffer_as_array_empty() -> None:
    buf = IncidentRateBuffer()
    arr = buf.as_array()
    assert len(arr) == 0


def test_incident_rate_buffer_as_array_values() -> None:
    buf = _make_buffer_with_data(10, base_count=5)
    arr = buf.as_array()
    assert len(arr) == 10
    assert all(v == 5.0 for v in arr)


def test_incident_rate_buffer_prunes_old_data() -> None:
    buf = IncidentRateBuffer(window_hours=5)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(10):
        buf.record(base + timedelta(hours=i), 1)
    assert len(buf.counts) <= 6


@pytest.mark.parametrize("window_hours", [24, 48, 168])
def test_incident_rate_buffer_window_hours_respected(window_hours: int) -> None:
    buf = IncidentRateBuffer(window_hours=window_hours)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(window_hours + 10):
        buf.record(base + timedelta(hours=i), 1)
    assert len(buf.counts) <= window_hours + 1


def test_incident_rate_buffer_get_counts_since() -> None:
    buf = _make_buffer_with_data(10)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    cutoff = base + timedelta(hours=5)
    result = buf.get_counts_since(cutoff)
    assert all(t >= cutoff for t, _ in result)


def test_exponential_smoothing_forecaster_not_fitted_raises() -> None:
    forecaster = ExponentialSmoothingForecaster()
    with pytest.raises(RuntimeError):
        base = datetime(2024, 1, 1, tzinfo=UTC)
        forecaster.forecast(base)


def test_exponential_smoothing_forecaster_fit_and_forecast() -> None:
    buf = _make_buffer_with_data(24)
    forecaster = ExponentialSmoothingForecaster(horizon=5)
    forecaster.fit(buf.as_array())
    base = datetime(2024, 1, 1, tzinfo=UTC)
    points = forecaster.forecast(base)
    assert len(points) == 5


@pytest.mark.parametrize("horizon", [1, 3, 6, 12, 24])
def test_exponential_smoothing_forecaster_horizon(horizon: int) -> None:
    buf = _make_buffer_with_data(24)
    forecaster = ExponentialSmoothingForecaster(horizon=horizon)
    forecaster.fit(buf.as_array())
    base = datetime(2024, 1, 1, tzinfo=UTC)
    points = forecaster.forecast(base)
    assert len(points) == horizon


def test_exponential_smoothing_forecaster_points_non_negative() -> None:
    buf = _make_buffer_with_data(24)
    forecaster = ExponentialSmoothingForecaster(horizon=10)
    forecaster.fit(buf.as_array())
    base = datetime(2024, 1, 1, tzinfo=UTC)
    points = forecaster.forecast(base)
    for pt in points:
        assert pt.value >= 0.0
        assert pt.lower_bound >= 0.0
        assert pt.upper_bound >= pt.lower_bound


def test_exponential_smoothing_forecaster_is_fitted() -> None:
    buf = _make_buffer_with_data(10)
    forecaster = ExponentialSmoothingForecaster()
    assert not forecaster.is_fitted
    forecaster.fit(buf.as_array())
    assert forecaster.is_fitted


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9])
def test_exponential_smoothing_forecaster_various_alpha(alpha: float) -> None:
    buf = _make_buffer_with_data(24)
    forecaster = ExponentialSmoothingForecaster(alpha=alpha, horizon=5)
    forecaster.fit(buf.as_array())
    base = datetime(2024, 1, 1, tzinfo=UTC)
    points = forecaster.forecast(base)
    assert len(points) == 5


def test_forecast_point_to_dict_has_required_keys() -> None:
    pt = ForecastPoint(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        value=5.0,
        lower_bound=3.0,
        upper_bound=7.0,
    )
    d = pt.to_dict()
    for key in ("timestamp", "value", "lower_bound", "upper_bound"):
        assert key in d


def test_forecast_to_dicts_serialises_all_points() -> None:
    buf = _make_buffer_with_data(24)
    forecaster = ExponentialSmoothingForecaster(horizon=3)
    forecaster.fit(buf.as_array())
    base = datetime(2024, 1, 1, tzinfo=UTC)
    points = forecaster.forecast(base)
    result = forecast_to_dicts(points)
    assert len(result) == 3
    assert all(isinstance(d, dict) for d in result)
