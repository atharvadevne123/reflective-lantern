"""Energy demand forecasting using simple statistical models."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ForecastMethod(str, Enum):
    """Demand forecasting methodology."""

    NAIVE = "naive"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_TREND = "linear_trend"


@dataclass
class ForecastResult:
    """Result of a demand forecast."""

    method: ForecastMethod
    forecast_values: list[float]
    mae: float
    rmse: float
    horizon: int


def naive_forecast(
    historical: list[float],
    horizon: int,
) -> list[float]:
    """Forecast using the last observed value (naive/persistence model).

    Args:
        historical: Historical demand values.
        horizon: Number of periods to forecast.

    Returns:
        List of forecasted values.

    Raises:
        ValueError: If historical is empty or horizon is non-positive.
    """
    if not historical:
        raise ValueError("historical must not be empty")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return [historical[-1]] * horizon


def moving_average_forecast(
    historical: list[float],
    window: int,
    horizon: int,
) -> list[float]:
    """Forecast using a simple moving average.

    Args:
        historical: Historical demand values.
        window: Number of periods in the moving average window.
        horizon: Number of periods to forecast.

    Returns:
        List of forecasted values.

    Raises:
        ValueError: If window exceeds history length, or arguments invalid.
    """
    if not historical:
        raise ValueError("historical must not be empty")
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(historical):
        raise ValueError("window must not exceed length of historical data")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    avg = sum(historical[-window:]) / window
    return [round(avg, 4)] * horizon


def exponential_smoothing_forecast(
    historical: list[float],
    alpha: float,
    horizon: int,
) -> list[float]:
    """Forecast using simple exponential smoothing (Holt's level model).

    Args:
        historical: Historical demand values.
        alpha: Smoothing factor in (0, 1].
        horizon: Number of periods to forecast.

    Returns:
        List of forecasted values.

    Raises:
        ValueError: If alpha is out of (0, 1] or history is empty.
    """
    if not historical:
        raise ValueError("historical must not be empty")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    level = historical[0]
    for value in historical[1:]:
        level = alpha * value + (1 - alpha) * level
    return [round(level, 4)] * horizon


def linear_trend_forecast(
    historical: list[float],
    horizon: int,
) -> list[float]:
    """Forecast using a fitted linear trend (OLS).

    Args:
        historical: Historical demand values.
        horizon: Number of periods to forecast.

    Returns:
        List of forecasted values.

    Raises:
        ValueError: If fewer than 2 historical values provided.
    """
    if len(historical) < 2:
        raise ValueError("At least 2 historical values required for linear trend")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    n = len(historical)
    x_mean = (n - 1) / 2.0
    y_mean = sum(historical) / n
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    ss_xy = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(historical))
    slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = y_mean - slope * x_mean
    return [round(intercept + slope * (n + h), 4) for h in range(horizon)]


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    """Compute mean absolute error between actual and predicted sequences.

    Args:
        actual: Actual observed values.
        predicted: Forecasted values.

    Returns:
        MAE as a float.

    Raises:
        ValueError: If lists are different lengths or empty.
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("sequences must not be empty")
    return round(sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual), 6)


def root_mean_square_error(actual: list[float], predicted: list[float]) -> float:
    """Compute root mean square error.

    Args:
        actual: Actual observed values.
        predicted: Forecasted values.

    Returns:
        RMSE as a float.

    Raises:
        ValueError: If lists are different lengths or empty.
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("sequences must not be empty")
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    return round(math.sqrt(mse), 6)


def forecast_demand(
    historical: list[float],
    horizon: int,
    method: ForecastMethod = ForecastMethod.MOVING_AVERAGE,
    window: int = 7,
    alpha: float = 0.3,
) -> ForecastResult:
    """Forecast future demand using a selected method and evaluate in-sample.

    Args:
        historical: Historical demand time series.
        horizon: Number of periods ahead to forecast.
        method: Forecasting method to use.
        window: Window size for moving average.
        alpha: Smoothing factor for exponential smoothing.

    Returns:
        Populated :class:`ForecastResult`.
    """
    if method == ForecastMethod.NAIVE:
        values = naive_forecast(historical, horizon)
    elif method == ForecastMethod.MOVING_AVERAGE:
        w = min(window, len(historical))
        values = moving_average_forecast(historical, w, horizon)
    elif method == ForecastMethod.EXPONENTIAL_SMOOTHING:
        values = exponential_smoothing_forecast(historical, alpha, horizon)
    else:
        values = linear_trend_forecast(historical, horizon)

    if len(historical) > 1:
        in_sample = [values[0]] * len(historical)
        mae = mean_absolute_error(historical, in_sample)
        rmse = root_mean_square_error(historical, in_sample)
    else:
        mae = 0.0
        rmse = 0.0

    return ForecastResult(
        method=method,
        forecast_values=values,
        mae=mae,
        rmse=rmse,
        horizon=horizon,
    )
