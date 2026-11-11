"""Simple time-series return forecasting utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_exponential_smoothing(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    """Single exponential smoothing for a price or return series."""
    result = np.empty(len(series))
    result[0] = series.iloc[0]
    for i in range(1, len(series)):
        result[i] = alpha * series.iloc[i] + (1 - alpha) * result[i - 1]
    return pd.Series(result, index=series.index)


def rolling_z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling Z-score for mean-reversion signal detection."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std().replace(0, np.nan)
    return (series - mu) / sigma


def forecast_next_return(prices: pd.Series, horizon: int = 5) -> dict[str, float]:
    """Naive momentum forecast: extrapolate recent average return."""
    returns = prices.pct_change().dropna()
    if len(returns) < 10:
        return {"forecast_return": 0.0, "lower_bound": -0.05, "upper_bound": 0.05}
    recent = returns.iloc[-20:]
    mu = float(recent.mean())
    sigma = float(recent.std())
    forecast = mu * horizon
    return {
        "forecast_return": round(forecast, 6),
        "lower_bound": round(forecast - 1.96 * sigma * np.sqrt(horizon), 6),
        "upper_bound": round(forecast + 1.96 * sigma * np.sqrt(horizon), 6),
    }
