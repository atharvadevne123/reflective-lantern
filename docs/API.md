# Traffic-Pulse API Documentation

Base URL: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs` (Swagger UI)

## Endpoints

### `GET /health`

Liveness probe.

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `ok` when the app is alive |
| `model_version` | string | Semantic version of the loaded model |
| `models_loaded` | string[] | Ensemble member names (`xgb`, `lgbm`) |

### `GET /api/v1/model-info`

Model metadata — version, ensemble members, the 26 feature names, and
congestion label mapping.

### `POST /api/v1/predict`

Single-route congestion prediction.

Required fields: `route_id`, `hour` (0-23), `day_of_week` (0-6),
`vehicle_count` (0-10000), `avg_speed_kmh` (0-200).

Optional fields: `month`, `road_type` (highway | arterial | collector |
local | expressway), `incident_count`, `temperature_celsius`, `is_raining`,
plus lag/rolling overrides (`lag_1h`, `lag_2h`, `lag_4h`, `rolling_mean_6h`,
`rolling_std_6h`, `rolling_mean_24h`).

Response: congestion level 0-3 with label, class probabilities, incident
score, and model version. Every prediction is persisted for monitoring.

### `POST /api/v1/predict/batch`

Same schema as `/predict` but accepts a JSON array (max 100 items) and
returns an array of predictions.

### `POST /api/v1/drift`

Two-sample Kolmogorov-Smirnov drift test.

Request: `feature_name` (string), `reference` (>=10 floats), `current`
(>=10 floats).

Response: `ks_statistic`, `p_value`, `drift_detected` (p < 0.05). Results
are persisted to `drift_logs` and gate the automated retraining pipeline.

### `GET /api/v1/routes/{route_id}/history`

Most recent logged predictions for one route segment. Query param `limit`
(default 20, clamped to 1-200).

### `GET /api/v1/metrics`

Monitoring snapshot: prediction volume, congestion level distribution,
active drift alerts, and training metrics (per-model CV AUC).

### `POST /api/v1/tariff/compare`

Prices an hourly load profile under flat, time-of-use, and tiered tariffs.

Request: JSON array of hourly kWh. Query param `start_hour` (0-23, default 0)
sets the clock hour of the first entry.

Response: `flat_cost`, `time_of_use_cost`, `tiered_cost`, `cheapest_scheme`,
`saving_vs_flat`, `hours_priced`.

### `POST /api/v1/load-profile`

Demand shape characterisation for one building.

Request: JSON array of hourly kWh.

Response: `base_load_kwh`, `peak_kwh`, `mean_kwh`, `load_factor`,
`peak_to_average`, `max_ramp_kwh`, and `profile_class` (`flat`, `moderate`,
or `peaky`).

### `POST /api/v1/weather-normalize`

Splits period-over-period consumption change into weather and efficiency
components using degree days.

Query params: `baseline_kwh`, `current_kwh`, `baseline_degree_days`,
`current_degree_days`.

Response: `raw_change_pct`, `normalized_change_pct`, `weather_effect_pct`,
`normalized_current_kwh`. The latter two sum to `raw_change_pct`.

### `POST /api/v1/demand-response/evaluate`

Settles a demand-response event against a baseline.

Request body: `baseline_hourly_kwh`, `actual_hourly_kwh` (equal lengths).
Query param `committed_kwh`.

Response: `curtailed_kwh`, `curtailment_pct`, `shortfall_kwh`, `incentive`,
`penalty`, `net_payment`, `performance_score` (0-1).

### `POST /api/v1/power-quality`

Supply-side power quality report.

Request: JSON array of phase voltages (at least 2). Query params
`real_power_kw`, `reactive_power_kvar`.

Response: `power_factor`, `power_factor_rating` (`good`/`acceptable`/`poor`),
`apparent_power_kva`, `voltage_imbalance_pct`, `imbalance_within_limit`.

### `GET /api/v1/power-quality/correction`

Capacitor rating needed to reach a target power factor.

Query params: `real_power_kw`, `current_power_factor`, `target_power_factor`
(default 0.95). Response: `required_kvar` (0.0 when already at target).

### `POST /api/v1/solar/economics`

Values on-site PV against site demand, matched hour by hour.

Request body: `generation_hourly_kwh`, `consumption_hourly_kwh` (equal
lengths). Query params `import_rate` (default 0.15), `export_rate` (0.05).

Response: `self_consumed_kwh`, `exported_kwh`, `imported_kwh`,
`self_consumption_rate`, `self_sufficiency_rate`, `bill_saving`,
`export_revenue`, `total_benefit`.

### `GET /api/v1/solar/payback`

Simple payback period for a PV system.

Query params: `system_cost`, `annual_benefit`, `annual_degradation`
(default 0.005). Response: `payback_years` (`null` when never repaid) and
`repays_within_lifetime`.

### `POST /api/v1/battery/peak-shave`

Simulates battery dispatch against an hourly load and values the reduction.

Request: JSON array of hourly kW. Query params `capacity_kwh`,
`max_charge_kw`, `max_discharge_kw`, `target_peak_kw`, and
`demand_charge_per_kw` (default 15.0).

Response: `peak_before_kw`, `peak_after_kw`, `peak_reduction_kw`,
`peak_reduction_pct`, `target_met`, `energy_discharged_kwh`,
`energy_charged_kwh`, `equivalent_cycles`, `demand_charge_saving`.

`target_met` is false when a power or capacity limit stopped the battery
from defending the target.

### `POST /api/v1/battery/sizing`

Usable capacity needed to hold load under a target, sized for the largest
single excursion.

Request: JSON array of hourly kW. Query param `target_peak_kw`.
Response: `required_usable_kwh`, `peak_load_kw`.

## Headers

Every response carries:

- `X-Correlation-ID` — echoed from the request or generated server-side.
- `X-Response-Time-Ms` — server-side processing time.

Rate limiting: 120 requests/minute per client IP (HTTP 429 beyond that,
with a `Retry-After: 60` header).
