"""Aftershock sequence forecasting using the modified Omori law.

The rate of aftershocks following a mainshock decays as a power law in time:

    n(t) = K / (t + c)^p

where ``t`` is time since the mainshock, ``p`` is the decay exponent (typically
0.9–1.5), ``c`` is a small time offset covering the incomplete-catalogue period
immediately after the rupture, and ``K`` scales with mainshock magnitude. This
module fits that law to an observed sequence and projects it forward.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_P = 1.1
DEFAULT_C = 0.05
# Bath's law: the largest aftershock averages ~1.2 magnitude units below the mainshock.
BATH_DELTA = 1.2


def omori_rate(t_days: float, k: float, p: float = DEFAULT_P, c: float = DEFAULT_C) -> float:
    """Aftershock rate (events/day) at ``t_days`` after the mainshock.

    Args:
        t_days: Time since the mainshock in days; negative values are clamped to 0.
        k: Productivity constant, scaling with mainshock magnitude.
        p: Omori decay exponent.
        c: Time offset in days, avoiding the singularity at t=0.

    Returns:
        Expected number of aftershocks per day at that moment.
    """
    t = max(0.0, t_days)
    return float(k / math.pow(t + c, p))


def expected_count(
    start_days: float,
    end_days: float,
    k: float,
    p: float = DEFAULT_P,
    c: float = DEFAULT_C,
) -> float:
    """Integrate the Omori rate over ``[start_days, end_days]``.

    Uses the closed-form integral of the power law rather than numeric
    quadrature, with the p=1 logarithmic case handled separately.

    Raises:
        ValueError: If the interval is inverted.
    """
    if end_days < start_days:
        raise ValueError("end_days must be >= start_days")

    lo, hi = max(0.0, start_days) + c, max(0.0, end_days) + c
    if math.isclose(p, 1.0):
        return float(k * (math.log(hi) - math.log(lo)))
    return float(k / (1.0 - p) * (math.pow(hi, 1.0 - p) - math.pow(lo, 1.0 - p)))


def productivity_from_magnitude(magnitude: float, alpha: float = 0.8) -> float:
    """Estimate the Omori productivity constant K from mainshock magnitude.

    K grows exponentially with magnitude; ``alpha`` is the productivity exponent.
    """
    return float(math.pow(10.0, alpha * (magnitude - 4.0)))


def fit_omori(
    times_days: list[float], p_grid: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5)
) -> dict[str, float]:
    """Fit K and p to observed aftershock times by maximum likelihood over a p grid.

    Args:
        times_days: Times of observed aftershocks, in days since the mainshock.
        p_grid: Candidate decay exponents to evaluate.

    Returns:
        Fitted ``k``, ``p``, ``c``, ``n_events`` and the achieved ``log_likelihood``.
        Falls back to defaults when there are too few events to fit.
    """
    times = sorted(t for t in times_days if t >= 0)
    if len(times) < 3:
        logger.debug("Too few aftershocks (%d) to fit Omori — using defaults", len(times))
        return {
            "k": float(max(1, len(times))),
            "p": DEFAULT_P,
            "c": DEFAULT_C,
            "n_events": len(times),
            "log_likelihood": 0.0,
            "fitted": False,
        }

    t_max = times[-1]
    best: dict[str, float] | None = None

    for p in p_grid:
        # For fixed p, the MLE of K is n / integral of the unit-rate law.
        unit_integral = expected_count(0.0, t_max, k=1.0, p=p, c=DEFAULT_C)
        if unit_integral <= 0:
            continue
        k = len(times) / unit_integral
        log_likelihood = float(
            sum(math.log(omori_rate(t, k=k, p=p, c=DEFAULT_C)) for t in times)
            - expected_count(0.0, t_max, k=k, p=p, c=DEFAULT_C)
        )
        if best is None or log_likelihood > best["log_likelihood"]:
            best = {
                "k": round(k, 4),
                "p": p,
                "c": DEFAULT_C,
                "n_events": len(times),
                "log_likelihood": round(log_likelihood, 4),
                "fitted": True,
            }

    return best or {
        "k": float(len(times)),
        "p": DEFAULT_P,
        "c": DEFAULT_C,
        "n_events": len(times),
        "log_likelihood": 0.0,
        "fitted": False,
    }


def forecast_sequence(
    mainshock_magnitude: float,
    horizon_days: int = 7,
    observed_times: list[float] | None = None,
) -> dict[str, Any]:
    """Forecast aftershock counts per day over ``horizon_days``.

    When ``observed_times`` is supplied the Omori parameters are fitted to that
    sequence; otherwise they are derived from the mainshock magnitude alone.

    Raises:
        ValueError: If ``horizon_days`` is not positive.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    if observed_times:
        params = fit_omori(observed_times)
    else:
        params = {
            "k": round(productivity_from_magnitude(mainshock_magnitude), 4),
            "p": DEFAULT_P,
            "c": DEFAULT_C,
            "n_events": 0,
            "log_likelihood": 0.0,
            "fitted": False,
        }

    daily = []
    for day in range(horizon_days):
        count = expected_count(
            float(day), float(day + 1), k=params["k"], p=params["p"], c=params["c"]
        )
        daily.append(
            {
                "day": day + 1,
                "expected_count": round(count, 3),
                "probability_at_least_one": round(1.0 - math.exp(-count), 4),
            }
        )

    total = sum(d["expected_count"] for d in daily)
    return {
        "mainshock_magnitude": round(mainshock_magnitude, 2),
        "horizon_days": horizon_days,
        "omori_parameters": params,
        "largest_expected_aftershock": round(max(0.0, mainshock_magnitude - BATH_DELTA), 2),
        "daily_forecast": daily,
        "total_expected": round(total, 3),
    }


def decay_half_life(k: float, p: float = DEFAULT_P, c: float = DEFAULT_C) -> float:
    """Days until the aftershock rate falls to half its value at t=0."""
    initial = omori_rate(0.0, k=k, p=p, c=c)
    if initial <= 0:
        return 0.0
    # Solve K/(t+c)^p = initial/2  ->  t = (2K/initial)^(1/p) - c
    return float(math.pow(2.0 * k / initial, 1.0 / p) - c)


def moving_average(values: list[float], window: int = 3) -> list[float]:
    """Simple trailing moving average, padded to preserve input length."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if not values:
        return []
    array = np.asarray(values, dtype=float)
    out = [float(array[max(0, i - window + 1) : i + 1].mean()) for i in range(len(array))]
    return [round(v, 4) for v in out]
