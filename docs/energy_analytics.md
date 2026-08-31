# Energy Analytics Modules

Reference for the domain modules that turn raw consumption readings into the
numbers an energy manager actually acts on: what a site pays, how its demand
is shaped, whether a change was real or just weather, and what on-site
generation or storage would be worth.

---

## `app.tariff` — Electricity pricing

Prices an hourly consumption series under three common tariff structures and
recommends the cheapest.

| Function | Purpose |
|----------|---------|
| `flat_rate_cost(hourly_kwh, rate)` | Single rate applied to total consumption |
| `time_of_use_cost(hourly_kwh, start_hour, ...)` | Peak/off-peak rates by clock hour |
| `tiered_cost(hourly_kwh, bands)` | Block pricing; each band charged at its own rate |
| `compare_tariffs(hourly_kwh, start_hour, flat_rate)` | Prices all three, names the cheapest |
| `peak_shift_saving(hourly_kwh, shiftable_fraction, ...)` | Value of moving load off-peak |

```python
from app.tariff import compare_tariffs

result = compare_tariffs([1.0] * 24, start_hour=0)
print(result.cheapest_scheme, result.saving_vs_flat)
```

Entries in `hourly_kwh` are assigned clock hours starting at `start_hour` and
wrapping at 24, so a series longer than a day is priced correctly.

**Endpoint:** `POST /api/v1/tariff/compare`

---

## `app.load_profile` — Demand shape

Reduces a consumption series to the shape metrics used to characterise a
building's demand.

| Function | Purpose |
|----------|---------|
| `base_load(hourly_kwh, percentile)` | Always-on load, read from the low tail |
| `load_factor(hourly_kwh)` | Mean ÷ peak; near 1.0 means flat and efficient |
| `peak_to_average_ratio(hourly_kwh)` | Peak ÷ mean; the reciprocal view |
| `max_ramp_rate(hourly_kwh)` | Largest hour-over-hour swing |
| `build_load_profile(hourly_kwh)` | All of the above, plus a profile class |

`classify_profile` buckets the load factor into `flat` (≥ 0.80),
`moderate` (≥ 0.45), or `peaky`.

**Endpoint:** `POST /api/v1/load-profile`

---

## `app.weather_normalization` — Weather-adjusted comparison

Separates weather-driven demand change from genuine efficiency change, so a
mild winter is not mistaken for a retrofit paying off.

| Function | Purpose |
|----------|---------|
| `heating_degree_days(temps, base)` | Σ max(0, base − temp) |
| `cooling_degree_days(temps, base)` | Σ max(0, temp − base) |
| `normalization_factor(baseline_dd, current_dd)` | Scaling ratio between periods |
| `normalize_consumption(kwh, baseline_dd, current_dd)` | Consumption at baseline weather |
| `compare_periods(...)` | Splits raw change into weather and efficiency parts |

`compare_periods` returns `raw_change_pct`, `normalized_change_pct`, and
`weather_effect_pct`; the latter two sum to the first. A negative raw change
paired with a positive normalized change is the case worth catching: usage
fell, but only because the weather was milder.

**Endpoint:** `POST /api/v1/weather-normalize`

---

## `app.demand_response` — Grid event settlement

Evaluates participation in a demand-response programme.

| Function | Purpose |
|----------|---------|
| `customer_baseline_load(history, days)` | Averages comparable days into an hourly CBL |
| `curtailment(baseline, actual)` | kWh reduced against the baseline |
| `performance_score(curtailed, committed)` | Delivery ÷ commitment, capped at 1.0 |
| `evaluate_event(...)` | Full settlement: incentive, penalty, net payment |

Curtailment is paid at the incentive rate; any shortfall against the
commitment is charged at the penalty rate. Over-delivery is paid in full but
does not create a credit.

**Endpoint:** `POST /api/v1/demand-response/evaluate`

---

## `app.power_quality` — Supply-side metrics

| Function | Purpose |
|----------|---------|
| `power_factor(real_kw, apparent_kva)` | Real ÷ apparent power |
| `apparent_power(real_kw, reactive_kvar)` | Vector sum of the two components |
| `reactive_power(real_kw, power_factor)` | Reactive component implied by a factor |
| `correction_kvar(real_kw, current_pf, target_pf)` | Capacitor rating to reach a target |
| `voltage_imbalance(phase_voltages)` | NEMA percentage imbalance across phases |
| `build_report(...)` | All of the above with pass/fail ratings |

Power factor is rated `good` (≥ 0.95), `acceptable` (≥ 0.85), or `poor`.
Voltage imbalance above 2% is flagged, per NEMA motor guidance.

**Endpoints:** `POST /api/v1/power-quality`, `GET /api/v1/power-quality/correction`

---

## `app.solar` — On-site generation

| Function | Purpose |
|----------|---------|
| `generation_kwh(area, irradiance, ...)` | PV output from array size and irradiance |
| `self_consumption(generation, consumption)` | Hourly split: self-used, exported, imported |
| `analyze_economics(...)` | Values that split against import and export rates |
| `payback_years(cost, benefit, degradation)` | Simple payback with annual degradation |

Self-consumption is matched **hour by hour**, not on daily totals — solar can
only offset load occurring at the same time, and a daily comparison would
overstate the benefit substantially.

`payback_years` returns `float('inf')` when the system never repays; the API
surfaces this as `null` with `repays_within_lifetime: false`.

**Endpoints:** `POST /api/v1/solar/economics`, `GET /api/v1/solar/payback`

---

## `app.battery` — Storage dispatch

| Function | Purpose |
|----------|---------|
| `BatterySpec(...)` | Capacity, power limits, efficiency, depth of discharge |
| `peak_shave(load, spec, target_peak_kw)` | Simulates dispatch against a demand ceiling |
| `required_capacity_kwh(load, target)` | Usable capacity to hold the target |
| `demand_charge_saving(result, tariff)` | Values the peak reduction |

The simulation starts the pack charged and applies round-trip losses **on the
way in**: storing 1 kWh draws `1 / efficiency` from the grid. `BatterySpec.usable_kwh`
applies the depth-of-discharge limit, so a 100 kWh pack at 80% DoD cycles 80 kWh.

When the battery cannot defend the target — because of a power limit or
insufficient capacity — `peak_after_kw` reports the peak actually reached and
a warning is logged, rather than silently reporting success.

**Endpoints:** `POST /api/v1/battery/peak-shave`, `POST /api/v1/battery/sizing`

---

## Error handling

Every function validates its inputs and raises `ValueError` with a message
naming the offending value. The API layer converts these to HTTP 422 with the
message in `detail`, so a bad request tells the caller exactly what was wrong.
