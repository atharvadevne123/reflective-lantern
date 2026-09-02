"""FastAPI application: /api/v1/predict, /api/v1/health, /api/v1/metrics,
/api/v1/anomaly, /api/v1/forecast, /api/v1/drift, /api/v1/batch-predict."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.model import generate_synthetic_data, get_metrics, load_model, predict, train_model
from app.monitoring import (
    check_all_features,
    compute_prediction_stats,
    detect_anomaly,
    log_anomaly,
    log_drift,
    log_prediction,
    set_reference_distributions,
    train_anomaly_detector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

_model_bundle: dict[str, Any] | None = None
_request_counts: dict[str, int] = {}
_rate_limit_window: dict[str, float] = {}
RATE_LIMIT = 300


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model_bundle
    init_db()
    _model_bundle = load_model()
    X_raw, y = generate_synthetic_data(500)
    set_reference_distributions({"consumption_kwh": y.tolist()})
    train_anomaly_detector(y.tolist())
    logger.info("Energy-Seer started — model loaded")
    yield
    logger.info("Energy-Seer shutting down")


app = FastAPI(
    title="Energy-Seer",
    description="Production ML API for energy consumption forecasting and smart grid anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next) -> Response:
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = _rate_limit_window.get(client_ip, now)
    if now - window_start > 60:
        _rate_limit_window[client_ip] = now
        _request_counts[client_ip] = 0
    _request_counts[client_ip] = _request_counts.get(client_ip, 0) + 1
    if _request_counts[client_ip] > RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    return await call_next(request)


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class EnergyReadingIn(BaseModel):
    meter_id: str = Field(..., min_length=1, max_length=64, description="Unique meter identifier")
    consumption_kwh: float = Field(..., ge=0, le=10000, description="Current consumption in kWh")
    temperature_c: float = Field(20.0, ge=-60, le=60, description="Ambient temperature Celsius")
    humidity_pct: float = Field(50.0, ge=0, le=100, description="Relative humidity percent")
    hour_of_day: int = Field(12, ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(0, ge=0, le=6, description="Day of week (0=Monday)")
    is_holiday: bool = Field(False, description="Whether the day is a public holiday")
    building_type: str = Field("residential", description="Building type category")

    @field_validator("building_type")
    @classmethod
    def validate_building_type(cls, v: str) -> str:
        allowed = {
            "residential",
            "commercial",
            "industrial",
            "data_center",
            "hospital",
            "school",
            "office",
            "retail",
        }
        if v.lower() not in allowed:
            raise ValueError(f"building_type must be one of {allowed}")
        return v.lower()


class PredictRequest(BaseModel):
    readings: list[EnergyReadingIn] = Field(..., min_length=1, max_length=100)
    horizon_h: int = Field(1, ge=1, le=168, description="Forecast horizon in hours")


class PredictResponse(BaseModel):
    predictions_kwh: list[float]
    horizon_h: int
    model_version: str
    meter_ids: list[str]


class AnomalyRequest(BaseModel):
    meter_id: str = Field(..., min_length=1, max_length=64)
    consumption_kwh: float = Field(..., ge=0, le=10000)


class DriftRequest(BaseModel):
    feature_values: dict[str, list[float]] = Field(
        ..., description="Map of feature name to list of recent values"
    )


class ForecastRequest(BaseModel):
    readings: list[EnergyReadingIn] = Field(..., min_length=1, max_length=24)
    steps: int = Field(24, ge=1, le=168, description="Number of hourly steps to forecast")


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/api/v1/health", tags=["System"])
async def health() -> dict:
    """Check API and model health status."""
    return {
        "status": "ok",
        "model_loaded": _model_bundle is not None,
        "version": "1.0.0",
    }


@app.get("/api/v1/metrics", tags=["System"])
async def metrics(db: Annotated[Session, Depends(get_db)]) -> dict:
    """Return model performance metrics and recent prediction statistics."""
    model_metrics = get_metrics()
    pred_stats = compute_prediction_stats(db)
    return {
        "model_metrics": model_metrics,
        "prediction_stats": pred_stats,
    }


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Forecast"])
async def predict_endpoint(
    req: PredictRequest,
    db: Annotated[Session, Depends(get_db)],
) -> PredictResponse:
    """
    Predict energy consumption for one or more meters.

    Returns forecasted kWh values for the requested horizon.
    """
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    readings_dicts = [r.model_dump() for r in req.readings]
    try:
        preds = predict(_model_bundle, readings_dicts, req.horizon_h)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    for reading, pred in zip(req.readings, preds, strict=False):
        log_prediction(
            db,
            meter_id=reading.meter_id,
            predicted_kwh=pred,
            features_used=reading.model_dump(),
            prediction_horizon_h=req.horizon_h,
        )

    return PredictResponse(
        predictions_kwh=preds,
        horizon_h=req.horizon_h,
        model_version="1.0.0",
        meter_ids=[r.meter_id for r in req.readings],
    )


@app.post("/api/v1/batch-predict", tags=["Forecast"])
async def batch_predict(
    req: PredictRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Batch prediction endpoint — processes up to 100 readings at once."""
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    readings_dicts = [r.model_dump() for r in req.readings]
    preds = predict(_model_bundle, readings_dicts, req.horizon_h)
    results = [
        {"meter_id": r.meter_id, "predicted_kwh": p}
        for r, p in zip(req.readings, preds, strict=False)
    ]
    return {"results": results, "count": len(results), "horizon_h": req.horizon_h}


@app.post("/api/v1/anomaly", tags=["Anomaly Detection"])
async def anomaly_endpoint(
    req: AnomalyRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """
    Detect anomalies in energy consumption using IsolationForest + Z-score.

    Returns anomaly score, severity, and type (spike/drop/none).
    """
    result = detect_anomaly(req.consumption_kwh)
    log_anomaly(db, req.meter_id, req.consumption_kwh, result)
    return {"meter_id": req.meter_id, "consumption_kwh": req.consumption_kwh, **result}


@app.post("/api/v1/forecast", tags=["Forecast"])
async def multi_step_forecast(
    req: ForecastRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """
    Generate multi-step hourly energy consumption forecast.

    Uses the provided seed readings and auto-increments hour_of_day for each step.
    """
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    forecasts = []
    base_readings = [r.model_dump() for r in req.readings]

    for step in range(req.steps):
        step_readings = []
        for r in base_readings:
            r_copy = dict(r)
            r_copy["hour_of_day"] = (r["hour_of_day"] + step) % 24
            r_copy["day_of_week"] = (r["day_of_week"] + (r["hour_of_day"] + step) // 24) % 7
            step_readings.append(r_copy)

        preds = predict(_model_bundle, step_readings, horizon_h=1)
        forecasts.append(
            {
                "step": step + 1,
                "hour_offset": step,
                "predictions_kwh": preds,
                "meter_ids": [r["meter_id"] for r in step_readings],
            }
        )

    return {"steps": req.steps, "forecast": forecasts}


@app.post("/api/v1/drift", tags=["Monitoring"])
async def drift_endpoint(
    req: DriftRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """
    Run KS-test drift detection on feature distributions.

    Returns per-feature drift statistics and flags drifted features.
    """
    results = check_all_features(req.feature_values)
    log_drift(db, results)
    drifted = [f for f, r in results.items() if r.get("drift_detected")]
    return {
        "feature_results": results,
        "drifted_features": drifted,
        "drift_detected": len(drifted) > 0,
        "total_features_checked": len(results),
    }


@app.post("/api/v1/retrain", tags=["System"], status_code=status.HTTP_202_ACCEPTED)
async def trigger_retrain() -> dict:
    """Trigger synchronous model retraining on synthetic data (demo endpoint)."""
    global _model_bundle
    _, metrics = train_model()
    _model_bundle = load_model()
    return {"status": "retrained", "metrics": metrics}
