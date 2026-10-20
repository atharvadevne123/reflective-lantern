"""Distributed Energy Resource (DER) management and grid interaction utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DERType(str, Enum):
    """Supported distributed energy resource types."""

    SOLAR_PV = "solar_pv"
    WIND = "wind"
    BATTERY = "battery"
    EV_CHARGER = "ev_charger"
    FUEL_CELL = "fuel_cell"
    CHP = "chp"
    DEMAND_RESPONSE = "demand_response"


class DERStatus(str, Enum):
    """Operational status of a DER unit."""

    ONLINE = "online"
    OFFLINE = "offline"
    STANDBY = "standby"
    FAULT = "fault"
    CURTAILED = "curtailed"


@dataclass
class DERUnit:
    """Represents a single DER asset."""

    asset_id: str
    der_type: DERType
    rated_power_kw: float
    current_output_kw: float = 0.0
    status: DERStatus = DERStatus.STANDBY
    location_id: str = ""
    priority: int = 5

    def __post_init__(self) -> None:
        if self.rated_power_kw <= 0:
            raise ValueError("rated_power_kw must be positive")
        if self.current_output_kw < 0:
            raise ValueError("current_output_kw must be non-negative")
        if not (1 <= self.priority <= 10):
            raise ValueError("priority must be in [1, 10]")

    @property
    def utilisation_pct(self) -> float:
        """Current utilisation as a percentage of rated power."""
        if self.rated_power_kw == 0:
            return 0.0
        return round(100.0 * self.current_output_kw / self.rated_power_kw, 2)

    @property
    def available_kw(self) -> float:
        """Available headroom in kW."""
        if self.status not in (DERStatus.ONLINE, DERStatus.STANDBY):
            return 0.0
        return round(max(0.0, self.rated_power_kw - self.current_output_kw), 4)


@dataclass
class DERPortfolio:
    """A collection of DER assets at a site or within a VPP."""

    assets: list[DERUnit] = field(default_factory=list)

    def total_rated_kw(self) -> float:
        """Sum of rated capacities across all assets."""
        return round(sum(a.rated_power_kw for a in self.assets), 4)

    def total_online_kw(self) -> float:
        """Sum of current output from all online assets."""
        return round(sum(a.current_output_kw for a in self.assets if a.status == DERStatus.ONLINE), 4)

    def available_kw(self) -> float:
        """Total dispatchable headroom across online/standby assets."""
        return round(sum(a.available_kw for a in self.assets), 4)

    def assets_by_type(self, der_type: DERType) -> list[DERUnit]:
        """Filter assets by DER type."""
        return [a for a in self.assets if a.der_type == der_type]

    def assets_online(self) -> list[DERUnit]:
        """Return only online assets."""
        return [a for a in self.assets if a.status == DERStatus.ONLINE]

    def utilisation_pct(self) -> float:
        """Portfolio-level utilisation percentage."""
        rated = self.total_rated_kw()
        if rated == 0:
            return 0.0
        return round(100.0 * self.total_online_kw() / rated, 2)


def dispatch_order(
    portfolio: DERPortfolio,
    required_kw: float,
) -> list[tuple[str, float]]:
    """Greedy dispatch: rank assets by priority and fill demand.

    Args:
        portfolio: DER portfolio to dispatch from.
        required_kw: Total power required from the portfolio in kW.

    Returns:
        List of (asset_id, dispatched_kw) tuples in dispatch order.

    Raises:
        ValueError: If required_kw < 0.
    """
    if required_kw < 0:
        raise ValueError("required_kw must be non-negative")

    dispatchable = sorted(
        [a for a in portfolio.assets if a.status in (DERStatus.ONLINE, DERStatus.STANDBY)],
        key=lambda a: a.priority,
    )
    dispatch: list[tuple[str, float]] = []
    remaining = required_kw
    for asset in dispatchable:
        if remaining <= 0:
            break
        output = min(asset.available_kw, remaining)
        if output > 0:
            dispatch.append((asset.asset_id, round(output, 4)))
            remaining -= output
    return dispatch


def aggregate_generation(portfolio: DERPortfolio, der_type: DERType | None = None) -> float:
    """Aggregate current output, optionally filtered by type.

    Args:
        portfolio: DER portfolio.
        der_type: Optional filter. If None, sum all online assets.

    Returns:
        Total current output in kW.
    """
    assets = portfolio.assets_by_type(der_type) if der_type else portfolio.assets
    return round(sum(a.current_output_kw for a in assets if a.status == DERStatus.ONLINE), 4)


def grid_import_export(
    site_load_kw: float,
    der_generation_kw: float,
) -> float:
    """Compute net grid import (+) or export (-).

    Args:
        site_load_kw: On-site electricity demand in kW.
        der_generation_kw: On-site DER generation in kW.

    Returns:
        Net grid interaction in kW. Positive = import, negative = export.

    Raises:
        ValueError: If either value is negative.
    """
    if site_load_kw < 0:
        raise ValueError("site_load_kw must be non-negative")
    if der_generation_kw < 0:
        raise ValueError("der_generation_kw must be non-negative")
    return round(site_load_kw - der_generation_kw, 4)


def self_consumption_rate(
    der_generation_kw: float,
    site_load_kw: float,
) -> float:
    """Fraction of DER generation consumed on-site.

    Args:
        der_generation_kw: Generation from DER assets in kW.
        site_load_kw: On-site demand in kW.

    Returns:
        Self-consumption rate in [0, 1].

    Raises:
        ValueError: If either value is negative.
    """
    if der_generation_kw < 0:
        raise ValueError("der_generation_kw must be non-negative")
    if site_load_kw < 0:
        raise ValueError("site_load_kw must be non-negative")
    if der_generation_kw == 0:
        return 0.0
    consumed = min(der_generation_kw, site_load_kw)
    return round(consumed / der_generation_kw, 6)


def self_sufficiency_rate(
    der_generation_kw: float,
    site_load_kw: float,
) -> float:
    """Fraction of site demand met by on-site DER.

    Args:
        der_generation_kw: Generation from DER in kW.
        site_load_kw: On-site demand in kW.

    Returns:
        Self-sufficiency rate in [0, 1].

    Raises:
        ValueError: If either value is negative.
    """
    if der_generation_kw < 0:
        raise ValueError("der_generation_kw must be non-negative")
    if site_load_kw < 0:
        raise ValueError("site_load_kw must be non-negative")
    if site_load_kw == 0:
        return 1.0
    return round(min(1.0, der_generation_kw / site_load_kw), 6)
