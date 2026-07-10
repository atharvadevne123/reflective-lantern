"""Volt-Cast FastAPI application — energy load prediction API."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.database import (
    DriftReport,
    PredictionLog,
    RetrainingLog,
    create_tables,
    get_db,
)
from app.features import build_raw_feature_df
from app.middleware import CorrelationIDMiddleware, RateLimitMiddleware
from app.model import (
    MODEL_VERSION,
    generate_synthetic_training_data,
    load_metrics,
    load_model,
    predict,
    train_model,
)
from app.monitoring import (
    compute_all_feature_drifts,
    get_tracker,
)
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DriftResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_model = None
_start_time = time.monotonic()
_REFERENCE_LOADS: list[float] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _REFERENCE_LOADS
    create_tables()
    _model = load_model()
    if _model is None:
        logger.info("no saved model found — training on synthetic data")
        X, y = generate_synthetic_training_data(n_samples=2000)
        _model, _ = train_model(X.values, y.values)
        _REFERENCE_LOADS = y.tolist()[:1000]
        logger.info("model trained and ready")
    else:
        X, y = generate_synthetic_training_data(n_samples=500)
        _REFERENCE_LOADS = y.tolist()[:500]
    yield
    logger.info("volt-cast shutting down")


app = FastAPI(
    title="Volt-Cast",
    description="Energy consumption and grid load prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(RateLimitMiddleware)


def _get_model():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    tracker = get_tracker()
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_version=MODEL_VERSION,
        prediction_count=tracker.count,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["System"])
async def metrics():
    """Model performance metrics."""
    m = load_metrics()
    tracker = get_tracker()
    lats = tracker.latencies
    p50 = float(np.percentile(lats, 50)) if lats else None
    p95 = float(np.percentile(lats, 95)) if lats else None
    return MetricsResponse(
        r2_mean=m.get("r2_mean"),
        r2_std=m.get("r2_std"),
        rmse_mean=m.get("rmse_mean"),
        rmse_std=m.get("rmse_std"),
        n_samples=m.get("n_samples"),
        n_features=m.get("n_features"),
        model_version=m.get("model_version", MODEL_VERSION),
        prediction_count=tracker.count,
        recent_latency_p50_ms=round(p50, 2) if p50 else None,
        recent_latency_p95_ms=round(p95, 2) if p95 else None,
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_load(
    req: PredictRequest,
    request=None,
    db: Session = Depends(get_db),
):
    """Predict energy load for a single time slot."""
    model = _get_model()
    t0 = time.monotonic()
    request_id = str(uuid.uuid4())

    df = build_raw_feature_df(
        hour=req.hour,
        day_of_week=req.day_of_week,
        month=req.month,
        is_weekend=req.is_weekend,
        temperature_c=req.temperature_c,
        humidity_pct=req.humidity_pct,
        historical_loads=req.historical_loads,
        region=req.region,
    )

    # Apply feature transformers manually for single-row inference
    row = _engineer_features(df).values
    pred = float(predict(model, row)[0])
    pred = max(500.0, min(pred, 15000.0))  # clamp to plausible MW range

    latency_ms = (time.monotonic() - t0) * 1000
    get_tracker().record(pred, latency_ms)

    db.add(PredictionLog(
        predicted_load_mw=pred,
        model_version=MODEL_VERSION,
        region=req.region,
        latency_ms=round(latency_ms, 2),
        request_id=request_id,
    ))
    db.commit()

    return PredictResponse(
        predicted_load_mw=round(pred, 2),
        model_version=MODEL_VERSION,
        region=req.region,
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/api/v1/batch-predict", response_model=BatchPredictResponse, tags=["Prediction"])
async def batch_predict(req: BatchPredictRequest, db: Session = Depends(get_db)):
    """Predict energy load for multiple time slots."""
    model = _get_model()
    t0 = time.monotonic()
    results = []

    for item in req.requests:
        t1 = time.monotonic()
        request_id = str(uuid.uuid4())
        df = build_raw_feature_df(
            hour=item.hour,
            day_of_week=item.day_of_week,
            month=item.month,
            is_weekend=item.is_weekend,
            temperature_c=item.temperature_c,
            humidity_pct=item.humidity_pct,
            historical_loads=item.historical_loads,
            region=item.region,
        )
        row = _engineer_features(df).values
        pred = float(predict(model, row)[0])
        pred = max(500.0, min(pred, 15000.0))
        latency_ms = (time.monotonic() - t1) * 1000
        get_tracker().record(pred, latency_ms)
        db.add(PredictionLog(
            predicted_load_mw=pred,
            model_version=MODEL_VERSION,
            region=item.region,
            latency_ms=round(latency_ms, 2),
            request_id=request_id,
        ))
        results.append(PredictResponse(
            predicted_load_mw=round(pred, 2),
            model_version=MODEL_VERSION,
            region=item.region,
            request_id=request_id,
            latency_ms=round(latency_ms, 2),
        ))

    db.commit()
    return BatchPredictResponse(
        predictions=results,
        total=len(results),
        batch_latency_ms=round((time.monotonic() - t0) * 1000, 2),
    )


@app.get("/api/v1/drift", response_model=DriftResponse, tags=["Monitoring"])
async def drift_report(db: Session = Depends(get_db)):
    """Return KS-test drift detection report for recent predictions."""
    tracker = get_tracker()
    current_loads = tracker.loads

    if len(current_loads) < 10:
        return DriftResponse(
            drifts=[],
            summary={"message": "insufficient_predictions", "count": len(current_loads)},
            model_version=MODEL_VERSION,
        )

    ref = _REFERENCE_LOADS[:500] if _REFERENCE_LOADS else current_loads[:200]
    drift_result = compute_all_feature_drifts(
        reference_df=_loads_to_df(ref),
        current_df=_loads_to_df(current_loads[-200:]),
        numeric_features=["load_mw"],
    )

    for d in drift_result:
        db.add(DriftReport(
            feature_name=d["feature"],
            ks_statistic=d["ks_statistic"],
            p_value=d["p_value"],
            drift_detected=d["drift_detected"],
            reference_window=len(ref),
            current_window=min(len(current_loads), 200),
            model_version=MODEL_VERSION,
        ))
    db.commit()

    n_drifted = sum(1 for d in drift_result if d["drift_detected"])
    return DriftResponse(
        drifts=drift_result,
        summary={
            "total_features": len(drift_result),
            "drifted": n_drifted,
            "drift_pct": round(100 * n_drifted / max(len(drift_result), 1), 1),
        },
        model_version=MODEL_VERSION,
    )


@app.get("/api/v1/forecast", response_model=ForecastResponse, tags=["Prediction"])
async def forecast(req: ForecastRequest = Depends()):
    """Generate a multi-hour load forecast."""
    model = _get_model()
    forecasts = []
    for h in range(req.horizon_hours):
        hour = (req.start_hour + h) % 24
        dow = (req.day_of_week + (req.start_hour + h) // 24) % 7
        df = build_raw_feature_df(
            hour=hour,
            day_of_week=dow,
            month=req.month,
            is_weekend=req.is_weekend,
            temperature_c=req.temperature_c,
            humidity_pct=req.humidity_pct,
            region=req.region,
        )
        row = _engineer_features(df).values
        pred = float(predict(model, row)[0])
        pred = max(500.0, min(pred, 15000.0))
        forecasts.append({"offset_hours": h, "hour": hour, "predicted_load_mw": round(pred, 2)})

    return ForecastResponse(
        region=req.region,
        horizon_hours=req.horizon_hours,
        forecasts=forecasts,
    )


@app.post("/api/v1/retrain", tags=["Administration"])
async def retrain(db: Session = Depends(get_db)):
    """Trigger model retraining on synthetic data (demo endpoint)."""
    global _model
    X, y = generate_synthetic_training_data(n_samples=3000)
    _model, metrics = train_model(X.values, y.values)
    db.add(RetrainingLog(
        trigger="api",
        samples_used=3000,
        r2_after=metrics.get("r2_mean"),
        rmse_after=metrics.get("rmse_mean"),
        status="success",
    ))
    db.commit()
    return {"status": "retrained", "metrics": metrics}


def _engineer_features(df) -> pd.DataFrame:
    """Apply the feature engineering steps inline for single-row inference."""
    import numpy as np

    from app.features import (
        DropColumnsTransformer,
        LagFeatureTransformer,
        PeakPeriodEncoder,
        RatioFeatureTransformer,
        RollingStatsTransformer,
        TemporalFeatureTransformer,
    )

    for step in [
        LagFeatureTransformer(),
        RollingStatsTransformer(),
        TemporalFeatureTransformer(),
        PeakPeriodEncoder(),
        RatioFeatureTransformer(),
        DropColumnsTransformer(),
    ]:
        df = step.fit_transform(df)

    return df.select_dtypes(include=[np.number])


def _loads_to_df(loads: list[float]) -> pd.DataFrame:
    import pandas as pd
    return pd.DataFrame({"load_mw": loads})
