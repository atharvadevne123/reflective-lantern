# Watt-Guard — Monitoring Guide

## Overview

Every prediction request is persisted to the `prediction_logs` table and checked for anomalies and distribution drift in real time.

## Drift Detection

Watt-Guard uses a two-sample Kolmogorov-Smirnov (KS) test to compare the **reference distribution** (the last 500 training samples) with the **current window** (recent predictions).

### Reference Window

Set automatically when the model is trained (`POST /api/v1/train`). The last `REFERENCE_WINDOW_SIZE` (default 500) consumption values are stored in memory. To reset it, retrain the model.

### Drift Check (`POST /api/v1/drift`)

```json
{
  "current_values": [12.1, 11.8, 13.4, ...]
}
```

Response:

```json
{
  "ks_statistic": 0.142,
  "p_value": 0.031,
  "drift_detected": true,
  "checked_features": ["consumption_kwh"]
}
```

A `p_value < 0.05` triggers `drift_detected: true`. Each check is logged to `drift_logs`.

### Thresholds

| Variable | Default | Effect |
|---|---|---|
| `DRIFT_P_THRESHOLD` | `0.05` | p-value below which drift is flagged |
| `REFERENCE_WINDOW_SIZE` | `500` | Max samples kept as reference |

## Anomaly Detection

The `IsolationForest` model scores each reading during `POST /api/v1/anomaly`. Scores below the contamination threshold are flagged.

### Severity Classification

| Flags | Severity |
|---|---|
| Both Z-score AND IQR | `critical` |
| Either Z-score OR IQR | `warning` |
| Neither | `none` |

### Anomaly Log

All anomaly checks are persisted to `anomaly_logs` regardless of outcome.

## Metrics Endpoint

`GET /api/v1/metrics` returns aggregate statistics:

```json
{
  "total_predictions": 1042,
  "total_anomalies_flagged": 17,
  "total_drift_events": 3,
  "reference_window_size": 500,
  "model_r2": 0.91,
  "model_mae_kwh": 0.87
}
```

## Database Schema

| Table | Key Columns |
|---|---|
| `prediction_logs` | `building_id`, `timestamp`, `predicted_kwh`, `actual_kwh`, `latency_ms` |
| `anomaly_logs` | `building_id`, `timestamp`, `consumption_kwh`, `anomaly_score`, `is_anomaly`, `severity` |
| `drift_logs` | `feature_name`, `ks_statistic`, `p_value`, `drift_detected`, `checked_at` |

## Alerting

Currently logging-only. To add alerting:
1. Subscribe to `drift_detected=true` rows in `drift_logs`
2. Or set up a cron to poll `GET /api/v1/metrics` and alert on `total_drift_events` increase

## Airflow Retraining

The `watt_guard_weekly_retrain` DAG (every Sunday) automatically resets the reference window after a successful retrain. Manual trigger:

```bash
airflow dags trigger watt_guard_weekly_retrain
```
