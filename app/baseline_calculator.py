"""Energy baseline calculation using ASHRAE Guideline 14 and IPMVP methods."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BaselineModel:
    """Simple energy baseline model parameters."""

    intercept: float
    slope: float
    r_squared: float
    baseline_kwh: float
    method: str


def mean_baseline(pre_retrofit_kwh: list[float]) -> float:
    """Compute a simple mean baseline from pre-retrofit consumption data.

    Args:
        pre_retrofit_kwh: List of historic consumption readings (kWh).

    Returns:
        Mean consumption as the baseline estimate.

    Raises:
        ValueError: If the list is empty.
    """
    if not pre_retrofit_kwh:
        raise ValueError("pre_retrofit_kwh must not be empty")
    return round(statistics.mean(pre_retrofit_kwh), 4)


def linear_regression(
    x: list[float],
    y: list[float],
) -> tuple[float, float, float]:
    """Fit a simple ordinary-least-squares regression y = a + b*x.

    Args:
        x: Independent variable (e.g. degree-days or occupancy hours).
        y: Dependent variable (e.g. energy consumption in kWh).

    Returns:
        Tuple of (intercept, slope, r_squared).

    Raises:
        ValueError: If x and y lengths differ or have fewer than 2 points.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 data points for regression")
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
    ss_xx = sum((xi - mean_x) ** 2 for xi in x)
    if ss_xx == 0.0:
        return (mean_y, 0.0, 0.0)
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    if ss_tot == 0.0:
        r_sq = 1.0
    else:
        ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y, strict=False))
        r_sq = max(0.0, 1.0 - ss_res / ss_tot)
    return (round(intercept, 6), round(slope, 6), round(r_sq, 6))


def adjusted_baseline(
    intercept: float,
    slope: float,
    reporting_period_x: float,
) -> float:
    """Compute adjusted baseline consumption for a reporting period.

    Args:
        intercept: Regression intercept (kWh at zero independent variable).
        slope: Regression slope (kWh per unit of independent variable).
        reporting_period_x: Independent variable value in the reporting period.

    Returns:
        Adjusted baseline consumption in kWh.
    """
    return round(intercept + slope * reporting_period_x, 4)


def energy_savings(
    adjusted_baseline_kwh: float,
    actual_kwh: float,
) -> float:
    """Calculate verified energy savings (IPMVP Option B approach).

    Args:
        adjusted_baseline_kwh: Modelled pre-retrofit consumption for the period.
        actual_kwh: Measured post-retrofit consumption.

    Returns:
        Savings in kWh; negative means consumption increased.
    """
    return round(adjusted_baseline_kwh - actual_kwh, 4)


def build_regression_baseline(
    historic_x: list[float],
    historic_kwh: list[float],
    reporting_x: float,
) -> BaselineModel:
    """Build a full regression-based baseline model.

    Args:
        historic_x: Independent variable series for the baseline period.
        historic_kwh: Energy consumption during the baseline period.
        reporting_x: Independent variable value in the reporting period.

    Returns:
        Populated :class:`BaselineModel`.
    """
    intercept, slope, r2 = linear_regression(historic_x, historic_kwh)
    baseline = adjusted_baseline(intercept, slope, reporting_x)
    return BaselineModel(
        intercept=intercept,
        slope=slope,
        r_squared=r2,
        baseline_kwh=baseline,
        method="OLS regression",
    )


def normalised_savings_percent(
    adjusted_baseline_kwh: float,
    actual_kwh: float,
) -> float:
    """Express savings as a percentage of the adjusted baseline.

    Args:
        adjusted_baseline_kwh: Adjusted baseline consumption.
        actual_kwh: Actual post-retrofit consumption.

    Returns:
        Savings percentage (positive = improvement, negative = regression).
        Returns 0.0 when baseline is zero.

    Raises:
        ValueError: If either argument is negative.
    """
    if adjusted_baseline_kwh < 0 or actual_kwh < 0:
        raise ValueError("Consumption values must be non-negative")
    if adjusted_baseline_kwh == 0.0:
        return 0.0
    savings = energy_savings(adjusted_baseline_kwh, actual_kwh)
    return round(100.0 * savings / adjusted_baseline_kwh, 4)
