"""FastAPI application for Quake-Net seismic prediction API."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import SeismicEvent, get_db, init_db
from app.model import load_model, predict_magnitude, read_champion_metrics
from app.monitoring import check_all_drift, compute_psi, get_store

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
)
logger = logging.getLogger("quake_net")

_model_cache: dict[str, Any] = {}
_counters: dict[str, int] = {"predictions": 0, "errors": 0, "drift_checks": 0}
_rate_limit: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW = 60.0
RATE_LIMIT_MAX = 200

FAULT_TYPES = ["strike_slip", "reverse", "normal", "oblique", "unknown"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _model_cache["pipeline"] = load_model()
    logger.info('"Quake-Net API ready"')
    yield
    logger.info('"Quake-Net API shutting down"')


app = FastAPI(
    title="Quake-Net",
    description="Seismic event magnitude prediction and aftershock forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    start = time.monotonic()
    response: Response = await call_next(request)
    elapsed = round((time.monotonic() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    logger.info(
        '"method":"%s","path":"%s","status":%d,"ms":%.2f,"cid":"%s"',
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        correlation_id,
    )
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ("/api/v1/health", "/"):
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _rate_limit.setdefault(client_ip, [])
    _rate_limit[client_ip] = [t for t in window if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit[client_ip]) >= RATE_LIMIT_MAX:
        return Response(
            content=json.dumps({"detail": "Rate limit exceeded"}),
            status_code=429,
            media_type="application/json",
        )
    _rate_limit[client_ip].append(now)
    return await call_next(request)


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Epicenter latitude in degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Epicenter longitude in degrees")
    depth_km: float = Field(..., gt=0.0, le=700.0, description="Focal depth in kilometres")
    station_count: int = Field(
        ...,
        ge=1,
        le=500,
        description="Number of seismograph stations recording the event",
    )
    p_wave_amplitude: float = Field(..., gt=0.0, description="P-wave peak amplitude (microns)")
    s_wave_amplitude: float = Field(..., gt=0.0, description="S-wave peak amplitude (microns)")
    epicentral_distance_km: float = Field(
        ...,
        gt=0.0,
        le=20000.0,
        description="Distance from epicenter to nearest station (km)",
    )
    fault_type: str = Field(..., description="Fault mechanism type")

    @field_validator("fault_type")
    @classmethod
    def validate_fault_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in FAULT_TYPES:
            raise ValueError(f"fault_type must be one of {FAULT_TYPES}")
        return v

    @field_validator("s_wave_amplitude")
    @classmethod
    def s_greater_than_p(cls, v: float, info) -> float:
        p = info.data.get("p_wave_amplitude", 0)
        if p > 0 and v < p * 0.5:
            raise ValueError("s_wave_amplitude is suspiciously small compared to p_wave_amplitude")
        return v


class PredictResponse(BaseModel):
    predicted_magnitude: float
    aftershock_probability: float
    magnitude_class: str
    model_version: str = "1.0.0"
    correlation_id: str = ""


class BatchPredictRequest(BaseModel):
    events: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    count: int
    errors: int


class DriftResponse(BaseModel):
    feature_drifts: list[dict]
    drift_detected_count: int
    total_features_checked: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    predictions_served: int
    version: str = "1.0.0"


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {"service": "Quake-Net", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/v1/health", response_model=HealthResponse, tags=["operations"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded="pipeline" in _model_cache,
        predictions_served=_counters["predictions"],
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(
    request: Request,
    body: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    pipeline = _model_cache.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = body.model_dump()
    try:
        result = predict_magnitude(pipeline, features)
    except Exception as exc:
        _counters["errors"] += 1
        logger.exception('"Prediction failed: %s"', exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    _counters["predictions"] += 1
    store = get_store()
    store.record(features, result["predicted_magnitude"])

    event = SeismicEvent(
        latitude=body.latitude,
        longitude=body.longitude,
        depth_km=body.depth_km,
        station_count=body.station_count,
        p_wave_amplitude=body.p_wave_amplitude,
        s_wave_amplitude=body.s_wave_amplitude,
        epicentral_distance_km=body.epicentral_distance_km,
        fault_type=body.fault_type,
        predicted_magnitude=result["predicted_magnitude"],
        aftershock_probability=result["aftershock_probability"],
    )
    db.add(event)
    db.commit()

    cid = getattr(request.state, "correlation_id", "")
    return PredictResponse(**result, correlation_id=cid)


@app.post("/api/v1/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
async def predict_batch(
    request: Request,
    body: BatchPredictRequest,
    db: Session = Depends(get_db),
) -> BatchPredictResponse:
    pipeline = _model_cache.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    error_count = 0
    cid = getattr(request.state, "correlation_id", "")
    store = get_store()

    for event_req in body.events:
        try:
            features = event_req.model_dump()
            result = predict_magnitude(pipeline, features)
            _counters["predictions"] += 1
            store.record(features, result["predicted_magnitude"])

            db.add(
                SeismicEvent(
                    latitude=event_req.latitude,
                    longitude=event_req.longitude,
                    depth_km=event_req.depth_km,
                    station_count=event_req.station_count,
                    p_wave_amplitude=event_req.p_wave_amplitude,
                    s_wave_amplitude=event_req.s_wave_amplitude,
                    epicentral_distance_km=event_req.epicentral_distance_km,
                    fault_type=event_req.fault_type,
                    predicted_magnitude=result["predicted_magnitude"],
                    aftershock_probability=result["aftershock_probability"],
                )
            )
            results.append(PredictResponse(**result, correlation_id=cid))
        except Exception as exc:
            error_count += 1
            logger.warning('"Batch prediction error: %s"', exc)

    db.commit()
    return BatchPredictResponse(results=results, count=len(results), errors=error_count)


@app.get("/api/v1/metrics", tags=["operations"])
async def metrics(db: Session = Depends(get_db)) -> dict:
    model_metrics = read_champion_metrics()
    return {
        "service_counters": _counters,
        "model_metrics": model_metrics,
        "rate_limit_per_min": RATE_LIMIT_MAX,
    }


@app.get("/api/v1/drift", response_model=DriftResponse, tags=["monitoring"])
async def drift_report(db: Session = Depends(get_db)) -> DriftResponse:
    _counters["drift_checks"] += 1
    drifts = check_all_drift(db_session=db)
    detected_count = sum(1 for d in drifts if d.get("drift_detected"))
    return DriftResponse(
        feature_drifts=drifts,
        drift_detected_count=detected_count,
        total_features_checked=len(drifts),
    )


@app.get("/api/v1/drift/psi", tags=["monitoring"])
async def drift_psi() -> dict:
    """Population Stability Index for top numeric features."""
    from app.monitoring import load_reference_distribution

    reference = load_reference_distribution()
    store = get_store()
    results = {}

    for feature in store.all_features():
        current = store.get_feature_window(feature)
        ref_data = reference.get(feature, [])
        if ref_data and current:
            results[feature] = {
                "psi": compute_psi(ref_data, current),
                "sample_size": len(current),
            }

    return {"psi_scores": results}


@app.post("/api/v1/similar", tags=["analysis"])
async def similar_events(
    body: PredictRequest,
    limit: int = 5,
    db: Session = Depends(get_db),
) -> dict:
    """Find historical events with the most similar seismic signature."""
    from app.similarity import get_index

    limit = max(1, min(limit, 25))
    rows = db.query(SeismicEvent).order_by(SeismicEvent.created_at.desc()).limit(500).all()
    if not rows:
        return {"matches": [], "count": 0, "note": "No historical events indexed yet"}

    records = [
        {
            "id": row.id,
            "depth_km": row.depth_km,
            "p_wave_amplitude": row.p_wave_amplitude,
            "s_wave_amplitude": row.s_wave_amplitude,
            "epicentral_distance_km": row.epicentral_distance_km,
            "station_count": row.station_count,
            "predicted_magnitude": row.predicted_magnitude,
            "fault_type": row.fault_type,
        }
        for row in rows
    ]
    index = get_index().build(records)
    matches = index.search(body.model_dump(), k=limit)
    return {"matches": matches, "count": len(matches), "indexed": index.size}


@app.get("/api/v1/anomalies", tags=["monitoring"])
async def anomalies(limit: int = 100, db: Session = Depends(get_db)) -> dict:
    """Flag unusual seismic signatures among recently logged events."""
    import pandas as pd

    from app.anomaly import SeismicAnomalyDetector, iqr_outliers, zscore_outliers

    limit = max(10, min(limit, 500))
    rows = db.query(SeismicEvent).order_by(SeismicEvent.created_at.desc()).limit(limit).all()
    if len(rows) < 10:
        return {
            "anomalies": [],
            "count": 0,
            "note": f"Need at least 10 events to score, have {len(rows)}",
        }

    frame = pd.DataFrame(
        [
            {
                "id": row.id,
                "depth_km": row.depth_km,
                "p_wave_amplitude": row.p_wave_amplitude,
                "s_wave_amplitude": row.s_wave_amplitude,
                "epicentral_distance_km": row.epicentral_distance_km,
                "station_count": row.station_count,
                "predicted_magnitude": row.predicted_magnitude,
            }
            for row in rows
        ]
    )

    scores = SeismicAnomalyDetector().fit(frame).score(frame)
    magnitudes = frame["predicted_magnitude"].tolist()
    z_flags = zscore_outliers(magnitudes)
    iqr_flags = iqr_outliers(magnitudes)

    flagged = [
        {
            "event_id": int(frame.at[i, "id"]),
            "predicted_magnitude": float(magnitudes[i]),
            "anomaly_score": scores[i]["anomaly_score"],
            "isolation_forest": scores[i]["is_anomaly"],
            "zscore_outlier": z_flags[i],
            "iqr_outlier": iqr_flags[i],
        }
        for i in range(len(frame))
        if scores[i]["is_anomaly"] or z_flags[i] or iqr_flags[i]
    ]
    return {"anomalies": flagged, "count": len(flagged), "events_scored": len(frame)}


@app.get("/api/v1/cache/stats", tags=["operations"])
async def cache_stats() -> dict:
    """Report TTL cache utilisation and hit rate."""
    from app.cache import get_cache

    return get_cache().stats()


@app.get("/api/v1/events/recent", tags=["data"])
async def recent_events(limit: int = 20, db: Session = Depends(get_db)) -> dict:
    limit = max(1, min(limit, 100))
    events = db.query(SeismicEvent).order_by(SeismicEvent.created_at.desc()).limit(limit).all()
    return {
        "events": [
            {
                "id": e.id,
                "predicted_magnitude": e.predicted_magnitude,
                "aftershock_probability": e.aftershock_probability,
                "latitude": e.latitude,
                "longitude": e.longitude,
                "depth_km": e.depth_km,
                "fault_type": e.fault_type,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
        "count": len(events),
    }
