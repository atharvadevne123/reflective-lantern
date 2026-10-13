"""Analytics endpoints for tariff, load-profile, solar, battery and benchmark analysis."""

from __future__ import annotations

import math
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel, model_validator

router = APIRouter(prefix="/api/v1")

__all__ = ["router"]


# ---------------------------------------------------------------------------
# Tariff Compare
# ---------------------------------------------------------------------------

FLAT_RATE = 0.15
TOU_PEAK_RATE = 0.30
TOU_OFF_RATE = 0.10
TOU_PEAK_START = 16
TOU_PEAK_END = 22
TIER1_LIMIT = 100.0
TIER1_RATE = 0.12
TIER2_RATE = 0.22


@router.post("/tariff/compare")
def tariff_compare(
    hourly_kwh: list[float],
    start_hour: Annotated[int, Query(ge=0, le=23)] = 0,
) -> dict[str, object]:
    """Compare flat-rate, time-of-use, and tiered tariff costs for a consumption series."""
    if not hourly_kwh:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="consumption series must not be empty")
    if any(v < 0 for v in hourly_kwh):
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="consumption values must be non-negative")

    n = len(hourly_kwh)
    flat_cost = sum(hourly_kwh) * FLAT_RATE

    tou_cost = 0.0
    for i, kwh in enumerate(hourly_kwh):
        hour = (start_hour + i) % 24
        rate = TOU_PEAK_RATE if TOU_PEAK_START <= hour < TOU_PEAK_END else TOU_OFF_RATE
        tou_cost += kwh * rate

    cumulative = 0.0
    tiered_cost = 0.0
    for kwh in hourly_kwh:
        remaining = kwh
        if cumulative < TIER1_LIMIT:
            in_tier1 = min(remaining, TIER1_LIMIT - cumulative)
            tiered_cost += in_tier1 * TIER1_RATE
            remaining -= in_tier1
            cumulative += in_tier1
        if remaining > 0:
            tiered_cost += remaining * TIER2_RATE
            cumulative += remaining

    cheapest = min(
        [("flat", flat_cost), ("time_of_use", tou_cost), ("tiered", tiered_cost)],
        key=lambda x: x[1],
    )[0]
    return {
        "flat_cost": round(flat_cost, 4),
        "time_of_use_cost": round(tou_cost, 4),
        "tiered_cost": round(tiered_cost, 4),
        "hours_priced": n,
        "cheapest_scheme": cheapest,
    }


# ---------------------------------------------------------------------------
# Load Profile
# ---------------------------------------------------------------------------


@router.post("/load-profile")
def load_profile(hourly_kwh: list[float]) -> dict[str, object]:
    """Compute load profile statistics: peak, base, mean, load factor, and ramp."""
    if not hourly_kwh:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="series must not be empty")

    n = len(hourly_kwh)
    peak = max(hourly_kwh)
    base = min(hourly_kwh)
    mean = sum(hourly_kwh) / n
    load_factor = mean / peak if peak > 0 else 1.0
    peak_to_average = peak / mean if mean > 0 else 0.0
    max_ramp = max(abs(hourly_kwh[i + 1] - hourly_kwh[i]) for i in range(n - 1)) if n > 1 else 0.0
    profile_class = "flat" if load_factor >= 0.9 else "peaky"
    return {
        "base_load_kwh": round(base, 4),
        "peak_kwh": round(peak, 4),
        "mean_kwh": round(mean, 4),
        "load_factor": round(load_factor, 4),
        "peak_to_average": round(peak_to_average, 4),
        "max_ramp_kwh": round(max_ramp, 4),
        "profile_class": profile_class,
    }


# ---------------------------------------------------------------------------
# Weather Normalize
# ---------------------------------------------------------------------------


@router.post("/weather-normalize")
def weather_normalize(
    baseline_kwh: Annotated[float, Query(gt=0)],
    current_kwh: Annotated[float, Query(ge=0)],
    baseline_degree_days: Annotated[float, Query(gt=0)],
    current_degree_days: Annotated[float, Query(gt=0)],
) -> dict[str, float]:
    """Normalize energy consumption change for heating/cooling degree-day differences."""
    raw_change_pct = (current_kwh - baseline_kwh) / baseline_kwh * 100.0
    weather_adjusted_current = current_kwh * (baseline_degree_days / current_degree_days)
    normalized_change_pct = (weather_adjusted_current - baseline_kwh) / baseline_kwh * 100.0
    weather_effect_pct = normalized_change_pct - raw_change_pct
    return {
        "raw_change_pct": round(raw_change_pct, 4),
        "normalized_change_pct": round(normalized_change_pct, 4),
        "weather_effect_pct": round(weather_effect_pct, 4),
    }


# ---------------------------------------------------------------------------
# Demand Response
# ---------------------------------------------------------------------------


class DRPayload(BaseModel):
    baseline_hourly_kwh: list[float]
    actual_hourly_kwh: list[float]

    @model_validator(mode="after")
    def check_lengths_match(self) -> DRPayload:
        if len(self.baseline_hourly_kwh) != len(self.actual_hourly_kwh):
            raise ValueError("baseline_hourly_kwh and actual_hourly_kwh must have the same length")
        return self


INCENTIVE_RATE = 0.10
PENALTY_RATE = 0.15


@router.post("/demand-response/evaluate")
def demand_response_evaluate(
    payload: DRPayload,
    committed_kwh: Annotated[float, Query(gt=0)],
) -> dict[str, float]:
    """Evaluate demand-response performance: curtailment, shortfall, incentive, and penalty."""
    total_baseline = sum(payload.baseline_hourly_kwh)
    total_actual = sum(payload.actual_hourly_kwh)
    curtailed_kwh = max(0.0, total_baseline - total_actual)
    shortfall_kwh = max(0.0, committed_kwh - curtailed_kwh)
    performance_score = min(1.0, curtailed_kwh / committed_kwh) if committed_kwh > 0 else 1.0
    incentive = round(committed_kwh * INCENTIVE_RATE, 2)
    penalty = round(shortfall_kwh * PENALTY_RATE, 2)
    net_payment = round(incentive - penalty, 2)
    return {
        "curtailed_kwh": round(curtailed_kwh, 4),
        "shortfall_kwh": round(shortfall_kwh, 4),
        "performance_score": round(performance_score, 4),
        "incentive": incentive,
        "penalty": penalty,
        "net_payment": net_payment,
    }


# ---------------------------------------------------------------------------
# Power Quality
# ---------------------------------------------------------------------------


def _power_factor(real_kw: float, reactive_kvar: float) -> float:
    apparent = math.sqrt(real_kw**2 + reactive_kvar**2)
    return real_kw / apparent if apparent > 0 else 1.0


@router.post("/power-quality")
def power_quality(
    phase_voltages: list[float],
    real_power_kw: Annotated[float, Query(gt=0)],
    reactive_power_kvar: Annotated[float, Query(ge=0)],
) -> dict[str, object]:
    """Return power factor and voltage imbalance metrics for a set of phase voltages."""
    if len(phase_voltages) < 2:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="at least 2 phase voltages are required")

    pf = _power_factor(real_power_kw, reactive_power_kvar)
    pf_rating = "good" if pf >= 0.90 else "poor"

    mean_v = sum(phase_voltages) / len(phase_voltages)
    imbalance = (max(phase_voltages) - min(phase_voltages)) / mean_v if mean_v > 0 else 0.0
    imbalance_within_limit = imbalance <= 0.05

    return {
        "power_factor": round(pf, 4),
        "power_factor_rating": pf_rating,
        "imbalance_pct": round(imbalance * 100, 2),
        "imbalance_within_limit": imbalance_within_limit,
    }


@router.get("/power-quality/correction")
def power_quality_correction(
    real_power_kw: Annotated[float, Query(gt=0)],
    current_power_factor: Annotated[float, Query(gt=0, le=1)],
    target_power_factor: Annotated[float, Query(gt=0, le=1)] = 0.95,
) -> dict[str, float | None]:
    """Calculate reactive power (kVAR) needed to reach a target power factor."""
    if current_power_factor >= target_power_factor:
        return {"required_kvar": 0.0}
    current_angle = math.acos(current_power_factor)
    target_angle = math.acos(target_power_factor)
    required_kvar = real_power_kw * (math.tan(current_angle) - math.tan(target_angle))
    return {"required_kvar": round(required_kvar, 4)}


# ---------------------------------------------------------------------------
# Solar
# ---------------------------------------------------------------------------


class SolarEconomicsPayload(BaseModel):
    generation_hourly_kwh: list[float]
    consumption_hourly_kwh: list[float]

    @model_validator(mode="after")
    def check_lengths_match(self) -> SolarEconomicsPayload:
        if len(self.generation_hourly_kwh) != len(self.consumption_hourly_kwh):
            raise ValueError(
                "generation_hourly_kwh and consumption_hourly_kwh must have the same length"
            )
        return self


IMPORT_RATE = 0.15
EXPORT_RATE = 0.08


@router.post("/solar/economics")
def solar_economics(payload: SolarEconomicsPayload) -> dict[str, float]:
    """Calculate solar self-consumption, export revenue, and total financial benefit."""
    self_consumed = sum(
        min(g, c)
        for g, c in zip(payload.generation_hourly_kwh, payload.consumption_hourly_kwh, strict=False)
    )
    exported = sum(
        max(0.0, g - c)
        for g, c in zip(payload.generation_hourly_kwh, payload.consumption_hourly_kwh, strict=False)
    )
    imported = sum(
        max(0.0, c - g)
        for g, c in zip(payload.generation_hourly_kwh, payload.consumption_hourly_kwh, strict=False)
    )
    generated = sum(payload.generation_hourly_kwh)
    consumed = sum(payload.consumption_hourly_kwh)
    bill_saving = round(self_consumed * IMPORT_RATE, 2)
    export_revenue = round(exported * EXPORT_RATE, 2)
    total_benefit = round(bill_saving + export_revenue, 2)
    return {
        "self_consumed_kwh": round(self_consumed, 4),
        "exported_kwh": round(exported, 4),
        "imported_kwh": round(imported, 4),
        "generated_kwh": round(generated, 4),
        "consumed_kwh": round(consumed, 4),
        "bill_saving": bill_saving,
        "export_revenue": export_revenue,
        "total_benefit": total_benefit,
    }


SOLAR_LIFETIME_YEARS = 25.0


@router.get("/solar/payback")
def solar_payback(
    system_cost: Annotated[float, Query(gt=0)],
    annual_benefit: Annotated[float, Query(ge=0)],
) -> dict[str, object]:
    """Calculate solar system payback period and whether it repays within its lifetime."""
    if annual_benefit <= 0:
        return {"repays_within_lifetime": False, "payback_years": None}
    payback = system_cost / annual_benefit
    repays = payback <= SOLAR_LIFETIME_YEARS
    return {"repays_within_lifetime": repays, "payback_years": round(payback, 2)}


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


@router.post("/battery/peak-shave")
def battery_peak_shave(
    hourly_load_kw: list[float],
    capacity_kwh: Annotated[float, Query(gt=0)],
    max_charge_kw: Annotated[float, Query(gt=0)],
    max_discharge_kw: Annotated[float, Query(gt=0)],
    target_peak_kw: Annotated[float, Query(ge=0)],
    demand_charge_per_kw: Annotated[float, Query(ge=0)] = 0.0,
) -> dict[str, object]:
    """Simulate battery peak shaving and return peak reduction and demand charge savings."""
    original_peak = max(hourly_load_kw) if hourly_load_kw else 0.0
    soc = capacity_kwh
    shaved = []
    for load in hourly_load_kw:
        if load > target_peak_kw:
            discharge = min(load - target_peak_kw, max_discharge_kw, soc)
            soc -= discharge
            shaved.append(load - discharge)
        else:
            charge = min(target_peak_kw - load, max_charge_kw, capacity_kwh - soc)
            soc += charge
            shaved.append(load)

    peak_after = max(shaved) if shaved else 0.0
    target_met = peak_after <= target_peak_kw + 1e-9
    peak_reduction = original_peak - peak_after
    demand_charge_saving = round(peak_reduction * demand_charge_per_kw, 2)
    return {
        "target_met": target_met,
        "peak_after_kw": round(peak_after, 4),
        "peak_reduction_kw": round(peak_reduction, 4),
        "demand_charge_saving": demand_charge_saving,
    }


@router.post("/battery/sizing")
def battery_sizing(
    hourly_load_kw: list[float],
    target_peak_kw: Annotated[float, Query(ge=0)],
) -> dict[str, float]:
    """Calculate the minimum usable battery capacity required to shave to a target peak."""
    peak_load = max(hourly_load_kw) if hourly_load_kw else 0.0
    required = sum(max(0.0, load - target_peak_kw) for load in hourly_load_kw)
    return {
        "required_usable_kwh": round(required, 4),
        "peak_load_kw": round(peak_load, 4),
    }


# ---------------------------------------------------------------------------
# Cohort Benchmark
# ---------------------------------------------------------------------------


def _grade(percentile: float) -> str:
    if percentile >= 80:
        return "A"
    if percentile >= 60:
        return "B"
    if percentile >= 40:
        return "C"
    if percentile >= 20:
        return "D"
    return "F"


@router.post("/benchmark/cohort")
def cohort_benchmark(
    cohort_eui: list[float],
    annual_kwh: Annotated[float, Query(gt=0)],
    floor_area_m2: Annotated[float, Query(gt=0)],
) -> dict[str, object]:
    """Benchmark a building's energy use intensity against a peer cohort and return a grade."""
    if len(cohort_eui) < 3:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="cohort must have at least 3 members")

    building_eui = annual_kwh / floor_area_m2
    n = len(cohort_eui)
    better_count = sum(1 for e in cohort_eui if e > building_eui)
    percentile_rank = round(better_count / n * 100.0, 2)

    sorted_eui = sorted(cohort_eui)
    mid = n // 2
    cohort_median = sorted_eui[mid] if n % 2 == 1 else (sorted_eui[mid - 1] + sorted_eui[mid]) / 2

    return {
        "grade": _grade(percentile_rank),
        "percentile_rank": percentile_rank,
        "building_eui": round(building_eui, 4),
        "cohort_size": n,
        "cohort_median_eui": round(cohort_median, 4),
    }
