"""FastAPI application for Temporal-Pulse: time-series anomaly detection and forecasting."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .anomaly import explain_anomaly
from .database import AnomalyEvent, Prediction, get_db, init_db
from .features import build_feature_matrix
from .model import (
    score_anomaly,
    train_anomaly_detector,
    train_forecaster,
    get_feature_importance,
    get_model_version,
)
from .monitoring import (
    get_metrics_summary,
    log_prediction,
    run_all_drift_tests,
    update_current_distribution,
)
from .schemas import (
    AnomalyResult,
    BatchAnomalyResponse,
    BatchSensorReadings,
    DriftReport,
    ForecastResult,
    HealthResponse,
    MetricsResponse,
    SensorReading,
    TrainRequest,
    TrainResponse,
    VersionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.7"))
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise database on startup."""
    logger.info("Starting Temporal-Pulse API v%s", API_VERSION)
    try:
        init_db()
    except Exception as e:
        logger.error("DB init failed: %s", e)
    yield
    logger.info("Temporal-Pulse API shutting down")


app = FastAPI(
    title="Temporal-Pulse",
    description="Multivariate Time-Series Anomaly Detection and Forecasting API",
    version=API_VERSION,
    lifespan=lifespan,
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 422 with sanitized errors (raw input may contain non-JSON floats like NaN)."""
    errors = [
        {"loc": list(err.get("loc", [])), "msg": err.get("msg"), "type": err.get("type")}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": errors})


CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Response:
    """Attach a correlation ID to every request and log latency."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Processing-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "method=%s path=%s status=%d latency_ms=%.2f corr_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        correlation_id,
    )
    return response


@app.get("/api/v1/health", response_model=HealthResponse, tags=["ops"])
async def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """Liveness and readiness check."""
    model_loaded = True
    db_connected = True
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_connected = False

    return HealthResponse(
        status="ok" if db_connected else "degraded",
        version=API_VERSION,
        model_loaded=model_loaded,
        db_connected=db_connected,
    )


@app.get("/api/v1/version", response_model=VersionResponse, tags=["ops"])
async def version() -> VersionResponse:
    """Return current API and model versions."""
    return VersionResponse(api_version=API_VERSION, model_version=get_model_version())


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["ops"])
async def metrics() -> MetricsResponse:
    """Return aggregated prediction and latency metrics."""
    summary = get_metrics_summary()
    return MetricsResponse(**summary)


@app.post("/api/v1/train", response_model=TrainResponse, tags=["model"])
async def train_model(request: TrainRequest, db: Session = Depends(get_db)) -> TrainResponse:
    """Train anomaly detector and forecaster on provided readings."""
    try:
        raw = [
            {"timestamp": r.timestamp, "sensor_id": r.sensor_id, **r.values}
            for r in request.readings
        ]
        df, feature_cols = build_feature_matrix(raw)
        X = df[feature_cols].to_numpy(dtype=np.float32)

        _, scaler = train_anomaly_detector(X, contamination=request.contamination)
        y = X[:, 0]
        _, metrics = train_forecaster(X, y)

        return TrainResponse(
            model_version=get_model_version(),
            samples_trained=len(X),
            features_used=len(feature_cols),
            metrics=metrics,
        )
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/detect", response_model=BatchAnomalyResponse, tags=["inference"])
async def detect_anomalies(
    batch: BatchSensorReadings,
    db: Session = Depends(get_db),
) -> BatchAnomalyResponse:
    """Detect anomalies in a batch of sensor readings."""
    t0 = time.perf_counter()
    try:
        raw = [
            {"timestamp": r.timestamp, "sensor_id": r.sensor_id, **r.values}
            for r in batch.readings
        ]
        df, feature_cols = build_feature_matrix(raw)
        X = df[feature_cols].to_numpy(dtype=np.float32)

        # Update serving distribution for drift tracking
        for col_idx, col in enumerate(feature_cols[:5]):
            update_current_distribution(col, X[:, col_idx].tolist())

        try:
            scores = score_anomaly(X)
        except RuntimeError:
            # Model not yet trained — use random scores as placeholder
            logger.warning("Model not trained; returning placeholder scores")
            scores = np.random.rand(len(X)).astype(np.float32)

        results = []
        anomaly_count = 0
        for i, reading in enumerate(batch.readings):
            score = float(scores[i])
            is_anom = score >= ANOMALY_THRESHOLD
            if is_anom:
                anomaly_count += 1
                db.add(
                    AnomalyEvent(
                        sensor_id=reading.sensor_id,
                        timestamp=reading.timestamp,
                        anomaly_score=score,
                        root_cause=str(explain_anomaly(X[i], feature_cols)["top_contributors"][:2]),
                        features_snapshot=reading.values,
                    )
                )

            results.append(
                AnomalyResult(
                    sensor_id=reading.sensor_id,
                    timestamp=reading.timestamp,
                    anomaly_score=score,
                    is_anomaly=is_anom,
                    threshold=ANOMALY_THRESHOLD,
                )
            )

            log_prediction(reading.sensor_id, score, [], (time.perf_counter() - t0) * 1000)

        try:
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("DB commit failed: %s", e)

        processing_ms = (time.perf_counter() - t0) * 1000
        return BatchAnomalyResponse(
            results=results,
            total_readings=len(batch.readings),
            anomaly_count=anomaly_count,
            processing_time_ms=processing_ms,
        )
    except Exception as e:
        logger.error("Detection failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/forecast", response_model=list[ForecastResult], tags=["inference"])
async def forecast(reading: SensorReading, horizon: int = 5) -> list[ForecastResult]:
    """Forecast future values for each channel of a sensor reading."""
    try:
        from .forecaster import multi_channel_forecast

        channel_series = {k: np.array([v], dtype=np.float32) for k, v in reading.values.items()}
        fc_results = multi_channel_forecast(channel_series, horizon=horizon)
        output = []
        for channel, fc in fc_results.items():
            if "error" in fc:
                continue
            output.append(
                ForecastResult(
                    sensor_id=reading.sensor_id,
                    channel=channel,
                    horizon=horizon,
                    predicted_values=fc.get("point", []),
                    confidence_lower=fc.get("lower", []),
                    confidence_upper=fc.get("upper", []),
                    model_version=get_model_version(),
                )
            )
        return output
    except Exception as e:
        logger.error("Forecast failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/drift", response_model=list[DriftReport], tags=["monitoring"])
async def drift_report() -> list[DriftReport]:
    """Run KS-test drift detection across all tracked features."""
    reports = run_all_drift_tests()
    return [DriftReport(**r) for r in reports]


@app.get("/api/v1/feature-importance", tags=["model"])
async def feature_importance() -> list[dict]:
    """Return top-20 feature importances from the forecaster."""
    return get_feature_importance()
