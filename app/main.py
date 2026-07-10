"""FastAPI application — Watt-Guard energy forecasting API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from app import __version__
from app.database import create_tables, get_db
from app.features import make_feature_row
from app.middleware import CorrelationIDMiddleware, RateLimitMiddleware
from app.model import (
    get_metrics,
    load_anomaly_model,
    load_model,
    predict,
    score_anomaly,
)
from app.monitoring import (
    LatencyTimer,
    compute_drift,
    get_prediction_stats,
    log_anomaly,
    log_prediction,
    set_reference_window,
)
from app.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    DriftRequest,
    DriftResponse,
    EnergyReadingIn,
    HealthResponse,
    MetricsResponse,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_model_bundle: dict[str, Any] | None = None
_anomaly_bundle: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_bundle, _anomaly_bundle
    create_tables()
    _model_bundle = load_model()
    _anomaly_bundle = load_anomaly_model()
    if _model_bundle:
        logger.info("Forecasting model loaded.")
    if _anomaly_bundle:
        logger.info("Anomaly model loaded.")
    yield


app = FastAPI(
    title="Watt-Guard",
    description="Smart building energy consumption forecasting and anomaly detection API.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(CorrelationIDMiddleware)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Return API liveness and model load status."""
    return HealthResponse(
        status="ok",
        model_loaded=_model_bundle is not None,
        anomaly_model_loaded=_anomaly_bundle is not None,
        version=__version__,
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict_consumption(
    payload: EnergyReadingIn,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Forecast energy consumption for a building at a given timestep."""
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Forecasting model not loaded. Run /api/v1/train first.")

    with LatencyTimer() as timer:
        row = make_feature_row(
            hour=payload.hour,
            day_of_week=payload.day_of_week,
            month=payload.month,
            temperature_c=payload.temperature_c,
            humidity_pct=payload.humidity_pct,
            occupancy=payload.occupancy,
            hvac_state=payload.hvac_state,
            consumption_kwh=payload.consumption_kwh,
        )
        kwh_pred = float(predict(_model_bundle, row)[0])

    log_prediction(
        db=db,
        building_id=payload.building_id,
        timestamp=payload.timestamp,
        predicted_kwh=kwh_pred,
        latency_ms=timer.ms,
    )
    return PredictResponse(
        building_id=payload.building_id,
        timestamp=payload.timestamp,
        predicted_kwh=round(kwh_pred, 3),
        model_version=__version__,
        latency_ms=timer.ms,
    )


@app.post("/api/v1/anomaly", response_model=AnomalyResponse, tags=["Anomaly Detection"])
def detect_anomaly(
    payload: AnomalyRequest,
    db: Session = Depends(get_db),
) -> AnomalyResponse:
    """Detect whether a consumption reading is anomalous."""
    if _anomaly_bundle is None:
        raise HTTPException(status_code=503, detail="Anomaly model not loaded. Run /api/v1/train first.")

    with LatencyTimer() as timer:
        row = make_feature_row(
            hour=payload.hour,
            day_of_week=payload.day_of_week,
            month=payload.month,
            temperature_c=payload.temperature_c,
            humidity_pct=payload.humidity_pct,
            occupancy=payload.occupancy,
            hvac_state=payload.hvac_state,
            consumption_kwh=payload.consumption_kwh,
        )
        result = score_anomaly(_anomaly_bundle, row)

    log_anomaly(
        db=db,
        building_id=payload.building_id,
        timestamp=payload.timestamp,
        consumption_kwh=payload.consumption_kwh,
        anomaly_score=result["anomaly_score"],
        is_anomaly=result["is_anomaly"],
        severity=result["severity"],
    )
    return AnomalyResponse(
        building_id=payload.building_id,
        timestamp=payload.timestamp,
        consumption_kwh=payload.consumption_kwh,
        anomaly_score=result["anomaly_score"],
        is_anomaly=bool(result["is_anomaly"]),
        severity=result["severity"],
        latency_ms=timer.ms,
    )


@app.post("/api/v1/drift", response_model=DriftResponse, tags=["Monitoring"])
def drift_check(payload: DriftRequest) -> DriftResponse:
    """Run KS-test between reference and current consumption distributions."""
    from app.monitoring import _reference_window

    ref = payload.reference_values if payload.reference_values else _reference_window
    if len(ref) < 10:
        raise HTTPException(status_code=400, detail="Reference distribution has fewer than 10 samples.")

    result = compute_drift(ref, payload.current_values)
    msg = "Drift detected — retraining recommended." if result["drift_detected"] else "No significant drift detected."
    return DriftResponse(
        ks_statistic=result["ks_statistic"],
        p_value=result["p_value"],
        drift_detected=result["drift_detected"],
        message=msg,
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Monitoring"])
def metrics(db: Session = Depends(get_db)) -> MetricsResponse:
    """Return prediction counts, anomaly counts, drift events, and model metrics."""
    stats = get_prediction_stats(db)
    return MetricsResponse(
        **stats,
        model_metrics=get_metrics(),
    )


@app.post("/api/v1/train", tags=["Training"])
def train_endpoint(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Train both forecasting and anomaly models on synthetic seed data."""
    global _model_bundle, _anomaly_bundle
    import numpy as np
    import pandas as pd

    from app.model import train_anomaly_model, train_model

    rng = np.random.default_rng(42)
    n = 2000
    hours = rng.integers(0, 24, n)
    dow = rng.integers(0, 7, n)
    months = rng.integers(1, 13, n)
    temp = rng.uniform(-5, 38, n)
    hum = rng.uniform(20, 90, n)
    occ = rng.integers(0, 200, n)
    hvac = rng.integers(0, 2, n)
    base = 10 + 0.3 * temp + 0.05 * occ + 5 * hvac
    consumption = base + rng.normal(0, 2, n)
    consumption = np.clip(consumption, 0, None)

    df = pd.DataFrame(
        {
            "hour": hours,
            "day_of_week": dow,
            "month": months,
            "temperature_c": temp,
            "humidity_pct": hum,
            "occupancy": occ,
            "hvac_state": hvac,
            "consumption_kwh": consumption,
        }
    )
    y = pd.Series(consumption)

    _model_bundle, metrics_out = train_model(df, y)
    _anomaly_bundle = train_anomaly_model(df)
    set_reference_window(consumption.tolist())
    return {"status": "trained", "metrics": metrics_out}
