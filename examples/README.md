# Examples

Quick-start curl commands for the Logistics-Flow API running locally.

## Prerequisites

```bash
# Start the server (requires a trained model at MODEL_PATH)
make run
# or: uvicorn app.main:app --reload --port 8000
```

---

## Health check

```bash
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
```

Expected: `{"status": "ok", ...}`

---

## Predict delivery time

```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "carrier": "Amazon",
    "distance_km": 8.0,
    "weight_kg": 0.8,
    "route_type": "urban",
    "hour_of_day": 13,
    "day_of_week": 3
  }' | python3 -m json.tool
```

---

## Batch prediction (up to 100 shipments)

```bash
curl -s -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "shipments": [
      {"carrier": "FedEx", "distance_km": 120.0, "weight_kg": 3.5,
       "route_type": "suburban", "hour_of_day": 9, "day_of_week": 1},
      {"carrier": "UPS",   "distance_km": 55.0,  "weight_kg": 1.2,
       "route_type": "urban",    "hour_of_day": 14, "day_of_week": 4}
    ]
  }' | python3 -m json.tool
```

---

## Model metrics

```bash
curl -s http://localhost:8000/api/v1/metrics | python3 -m json.tool
```

---

## Drift status

```bash
curl -s http://localhost:8000/api/v1/drift | python3 -m json.tool
```

Returns `{"drift_detected": false, ...}` when the feature distribution is
stable relative to the training reference window.

---

## Passing a correlation ID

All responses echo the `X-Request-ID` header for distributed tracing:

```bash
curl -s http://localhost:8000/api/v1/health \
  -H "X-Request-ID: my-trace-id" \
  -v 2>&1 | grep "X-Request-ID"
```
