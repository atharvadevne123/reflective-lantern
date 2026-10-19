"""Microgrid energy management and dispatch controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DispatchMode(str, Enum):
    """Microgrid dispatch operating modes."""

    GRID_CONNECTED = "grid_connected"
    ISLAND_MODE = "island_mode"
    PEAK_SHAVING = "peak_shaving"
    SELF_CONSUMPTION = "self_consumption"


@dataclass
class MicrogridState:
    """Snapshot of microgrid operating state."""

    pv_generation_kw: float
    battery_soc_pct: float
    grid_import_kw: float
    grid_export_kw: float
    load_kw: float
    mode: DispatchMode


@dataclass
class DispatchDecision:
    """Dispatch controller output for a single time step."""

    battery_charge_kw: float
    battery_discharge_kw: float
    grid_import_kw: float
    grid_export_kw: float
    curtailed_pv_kw: float
    unmet_load_kw: float


def net_load_kw(load_kw: float, pv_kw: float) -> float:
    """Compute net load after subtracting PV generation.

    Args:
        load_kw: Electrical demand in kW.
        pv_kw: PV generation in kW.

    Returns:
        Net load in kW (negative means surplus PV).

    Raises:
        ValueError: If any argument is negative.
    """
    if load_kw < 0:
        raise ValueError("load_kw must be non-negative")
    if pv_kw < 0:
        raise ValueError("pv_kw must be non-negative")
    return round(load_kw - pv_kw, 4)


def self_consumption_ratio(
    load_kwh: float,
    pv_kwh: float,
    exported_kwh: float,
) -> float:
    """Compute PV self-consumption ratio.

    Args:
        load_kwh: Total load energy in kWh.
        pv_kwh: Total PV generation in kWh.
        exported_kwh: PV energy exported to grid in kWh.

    Returns:
        Self-consumption ratio as a fraction (0–1).

    Raises:
        ValueError: If pv_kwh is non-positive or exported > pv.
    """
    if pv_kwh <= 0:
        raise ValueError("pv_kwh must be positive")
    if exported_kwh < 0:
        raise ValueError("exported_kwh must be non-negative")
    if exported_kwh > pv_kwh:
        raise ValueError("exported_kwh cannot exceed pv_kwh")
    return round((pv_kwh - exported_kwh) / pv_kwh, 6)


def self_sufficiency_ratio(
    load_kwh: float,
    imported_kwh: float,
) -> float:
    """Compute site self-sufficiency ratio.

    Args:
        load_kwh: Total site load in kWh.
        imported_kwh: Energy imported from grid in kWh.

    Returns:
        Self-sufficiency ratio as a fraction (0–1).

    Raises:
        ValueError: If load_kwh is non-positive or imported exceeds load.
    """
    if load_kwh <= 0:
        raise ValueError("load_kwh must be positive")
    if imported_kwh < 0:
        raise ValueError("imported_kwh must be non-negative")
    return round(max(0.0, (load_kwh - imported_kwh) / load_kwh), 6)


def dispatch_self_consumption(
    load_kw: float,
    pv_kw: float,
    battery_soc_pct: float,
    battery_capacity_kwh: float,
    max_charge_kw: float,
    max_discharge_kw: float,
    min_soc_pct: float = 20.0,
    max_soc_pct: float = 90.0,
) -> DispatchDecision:
    """Dispatch controller for self-consumption maximisation.

    Args:
        load_kw: Current electrical demand in kW.
        pv_kw: Current PV generation in kW.
        battery_soc_pct: Battery state of charge in %.
        battery_capacity_kwh: Battery capacity in kWh.
        max_charge_kw: Maximum charge power in kW.
        max_discharge_kw: Maximum discharge power in kW.
        min_soc_pct: Minimum allowed SOC in %.
        max_soc_pct: Maximum allowed SOC in %.

    Returns:
        Populated :class:`DispatchDecision`.

    Raises:
        ValueError: If battery SOC or power values are invalid.
    """
    if not (0 <= battery_soc_pct <= 100):
        raise ValueError("battery_soc_pct must be in [0, 100]")
    if battery_capacity_kwh <= 0:
        raise ValueError("battery_capacity_kwh must be positive")
    if max_charge_kw < 0 or max_discharge_kw < 0:
        raise ValueError("Power limits must be non-negative")

    surplus = pv_kw - load_kw
    charge_kw = 0.0
    discharge_kw = 0.0
    grid_import = 0.0
    grid_export = 0.0
    curtailed = 0.0
    unmet = 0.0

    if surplus > 0:
        headroom = (max_soc_pct - battery_soc_pct) / 100.0 * battery_capacity_kwh
        charge_kw = min(surplus, max_charge_kw, headroom)
        remaining_surplus = surplus - charge_kw
        if remaining_surplus > 0:
            grid_export = remaining_surplus
    else:
        deficit = -surplus
        available = (battery_soc_pct - min_soc_pct) / 100.0 * battery_capacity_kwh
        discharge_kw = min(deficit, max_discharge_kw, max(0.0, available))
        remaining_deficit = deficit - discharge_kw
        if remaining_deficit > 0:
            grid_import = remaining_deficit

    return DispatchDecision(
        battery_charge_kw=round(charge_kw, 4),
        battery_discharge_kw=round(discharge_kw, 4),
        grid_import_kw=round(grid_import, 4),
        grid_export_kw=round(grid_export, 4),
        curtailed_pv_kw=round(curtailed, 4),
        unmet_load_kw=round(unmet, 4),
    )
