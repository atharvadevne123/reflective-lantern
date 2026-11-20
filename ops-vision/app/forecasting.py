"""Time-series incident rate forecasting using exponential smoothing."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ForecastPoint:
    """A single forecasted value at a future timestamp.

    Attributes:
        timestamp: The datetime being forecasted.
        value: Predicted incident rate (incidents per hour).
        lower_bound: 80% prediction interval lower bound.
        upper_bound: 80% prediction interval upper bound.
    """

    timestamp: datetime
    value: float
    lower_bound: float
    upper_bound: float


@dataclass
class IncidentRateBuffer:
    """Ring buffer accumulating hourly incident counts for forecasting.

    Attributes:
        window_hours: Number of past hours to retain for model fitting.
        counts: Ordered list of (timestamp, incident_count) tuples.
    """

    window_hours: int = 168
    counts: list[tuple[datetime, int]] = field(default_factory=list)

    def record(self, ts: datetime, count: int) -> None:
        """Append a new hourly count and prune entries outside the window.

        Args:
            ts: Timestamp of the observation.
            count: Number of incidents in that hour.
        """
        self.counts.append((ts, count))
        cutoff = ts - timedelta(hours=self.window_hours)
        self.counts = [(t, c) for t, c in self.counts if t >= cutoff]

    def as_array(self) -> np.ndarray:
        """Return counts as a float64 numpy array ordered by timestamp."""
        if not self.counts:
            return np.array([], dtype=np.float64)
        sorted_counts = sorted(self.counts, key=lambda x: x[0])
        return np.array([c for _, c in sorted_counts], dtype=np.float64)


class ExponentialSmoothingForecaster:
    """Simple double exponential smoothing (Holt's linear) forecaster.

    Produces horizon-step-ahead forecasts with a symmetric prediction interval
    derived from the residual standard deviation.

    Attributes:
        alpha: Level smoothing parameter in (0, 1).
        beta: Trend smoothing parameter in (0, 1).
        horizon: Number of steps ahead to forecast.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        horizon: int = 24,
    ) -> None:
        """Initialise the forecaster.

        Args:
            alpha: Level smoothing factor.
            beta: Trend smoothing factor.
            horizon: Number of future steps to project.
        """
        self.alpha = alpha
        self.beta = beta
        self.horizon = horizon
        self._level: Optional[float] = None
        self._trend: Optional[float] = None
        self._residuals: list[float] = []

    def fit(self, series: np.ndarray) -> "ExponentialSmoothingForecaster":
        """Fit the model on a time series.

        Args:
            series: 1-D array of historical values.

        Returns:
            Self (for method chaining).
        """
        if len(series) < 2:
            logger.warning("Series too short (%d pts) — using naive forecast", len(series))
            self._level = float(series[-1]) if len(series) == 1 else 0.0
            self._trend = 0.0
            return self

        level = series[0]
        trend = series[1] - series[0]
        residuals: list[float] = []

        for obs in series[1:]:
            prev_level = level
            level = self.alpha * obs + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
            residuals.append(float(obs - (prev_level + trend)))

        self._level = float(level)
        self._trend = float(trend)
        self._residuals = residuals
        logger.debug(
            "Holt fit: level=%.4f trend=%.4f on %d points", level, trend, len(series)
        )
        return self

    def forecast(
        self, base_time: datetime, step_hours: int = 1
    ) -> list[ForecastPoint]:
        """Generate horizon-step-ahead forecasts.

        Args:
            base_time: The datetime corresponding to the last observed point.
            step_hours: Interval between forecast points in hours.

        Returns:
            List of ForecastPoint covering the next horizon steps.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._level is None or self._trend is None:
            raise RuntimeError("Forecaster must be fitted before calling forecast()")

        residual_std = (
            float(np.std(self._residuals)) if self._residuals else 1.0
        )
        z80 = 1.282

        points: list[ForecastPoint] = []
        for h in range(1, self.horizon + 1):
            value = max(0.0, self._level + h * self._trend)
            interval = z80 * residual_std * (h ** 0.5)
            lower = max(0.0, value - interval)
            upper = value + interval
            ts = base_time + timedelta(hours=h * step_hours)
            points.append(
                ForecastPoint(
                    timestamp=ts,
                    value=round(value, 4),
                    lower_bound=round(lower, 4),
                    upper_bound=round(upper, 4),
                )
            )

        logger.info(
            "Forecast generated: %d points, next=%.4f", len(points), points[0].value
        )
        return points


_buffer_singleton: Optional[IncidentRateBuffer] = None


def get_rate_buffer() -> IncidentRateBuffer:
    """Return the global IncidentRateBuffer singleton."""
    global _buffer_singleton
    if _buffer_singleton is None:
        _buffer_singleton = IncidentRateBuffer()
    return _buffer_singleton
