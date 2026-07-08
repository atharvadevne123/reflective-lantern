"""FastAPI application for Realty-Edge property valuation API."""

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.faiss_index import add_property, search_comparable
from app.model import MODEL_VERSION, load_metrics, load_model, predict
from app.monitoring import (
    get_drift_summary,
    get_recent_predictions,
    log_prediction,
    run_drift_check,
)
from app.schemas import (
    BatchPredictionResponse,
    BatchPropertyInput,
    ComparableRequest,
    ComparableResponse,
    DriftStatusResponse,
    HealthResponse,
    MetricsResponse,
    NeighborhoodStatsResponse,
    PredictionResponse,
    PropertyInput,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["300/minute"])

_model_bundle: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model_bundle
    init_db()
    _model_bundle = load_model()
    logger.info("Realty-Edge started. Model version=%s", MODEL_VERSION)
    yield
    logger.info("Realty-Edge shutting down.")


app = FastAPI(
    title="Realty-Edge",
    description="Real estate property valuation and investment scoring API.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Response:
    corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = corr_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Correlation-ID"] = corr_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
    return response


def _property_to_df(prop: PropertyInput) -> pd.DataFrame:
    return pd.DataFrame([prop.model_dump()])


def _investment_score(predicted_value: float, prop: PropertyInput) -> float:
    if predicted_value <= 0:
        return 0.0
    annual_rent = predicted_value * prop.avg_rental_yield
    cap_rate = annual_rent / predicted_value
    amenity = (
        prop.school_score * 0.4 + prop.transit_score * 0.3 + prop.walkability_score * 0.3
    ) / 10
    risk_penalty = prop.crime_rate
    score = (cap_rate * 50 + amenity * 3 - risk_penalty * 2) * 10
    return min(max(round(float(score), 2), 0.0), 10.0)


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns service status, model version, and DB connectivity.",
)
async def health(db: Session = Depends(get_db)) -> HealthResponse:
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return HealthResponse(status="ok", model_version=MODEL_VERSION, db_connected=db_ok)


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    tags=["System"],
    summary="Model metrics",
    description="Returns 5-fold CV R2, RMSE, feature count from the last training run.",
)
async def metrics() -> MetricsResponse:
    m = load_metrics()
    return MetricsResponse(**{k: v for k, v in m.items() if k in MetricsResponse.model_fields})


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Valuation"],
    summary="Predict property value",
    description="Predict market value and investment score for a single property.",
)
@limiter.limit("100/minute")
async def predict_value(
    request: Request,
    prop: PropertyInput,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    corr_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = _property_to_df(prop)
    try:
        values = predict(df, _model_bundle)
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    val = float(values[0])
    band_low = val * 0.92
    band_high = val * 1.08
    inv_score = _investment_score(val, prop)

    log_prediction(
        db=db,
        predicted_value=val,
        features=prop.model_dump(),
        correlation_id=corr_id,
        investment_score=inv_score,
    )
    add_property(
        np.array(
            [
                prop.sqft,
                prop.bedrooms,
                prop.bathrooms,
                prop.condition_score,
                prop.school_score,
                prop.transit_score,
                prop.walkability_score,
                prop.crime_rate,
                prop.median_price_per_sqft,
                prop.avg_rental_yield,
                prop.listing_days,
                float(prop.year_built),
            ]
            + [0.0] * 12
        ),
        {"predicted_value": val, "zipcode": prop.zipcode, "sqft": prop.sqft},
    )
    return PredictionResponse(
        predicted_value=round(val, 2),
        investment_score=inv_score,
        confidence_band_low=round(band_low, 2),
        confidence_band_high=round(band_high, 2),
        model_version=MODEL_VERSION,
        correlation_id=corr_id,
    )


@app.post(
    "/api/v1/batch-predict",
    response_model=BatchPredictionResponse,
    tags=["Valuation"],
    summary="Batch predict",
    description="Predict values for up to 100 properties in one request.",
)
@limiter.limit("20/minute")
async def batch_predict(
    request: Request,
    body: BatchPropertyInput,
    db: Session = Depends(get_db),
) -> BatchPredictionResponse:
    corr_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([p.model_dump() for p in body.properties])
    try:
        values = predict(df, _model_bundle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc

    results = []
    for val, prop in zip(values, body.properties, strict=False):
        v = float(val)
        inv = _investment_score(v, prop)
        results.append(
            PredictionResponse(
                predicted_value=round(v, 2),
                investment_score=inv,
                confidence_band_low=round(v * 0.92, 2),
                confidence_band_high=round(v * 1.08, 2),
                model_version=MODEL_VERSION,
                correlation_id=corr_id,
            )
        )
    return BatchPredictionResponse(predictions=results, count=len(results))


@app.post("/api/v1/comparable-properties", response_model=ComparableResponse, tags=["Search"])
@limiter.limit("50/minute")
async def comparable_properties(request: Request, body: ComparableRequest) -> ComparableResponse:
    prop = body.property
    query_vec = np.array(
        [
            prop.sqft,
            prop.bedrooms,
            prop.bathrooms,
            prop.condition_score,
            prop.school_score,
            prop.transit_score,
            prop.walkability_score,
            prop.crime_rate,
            prop.median_price_per_sqft,
            prop.avg_rental_yield,
            prop.listing_days,
            float(prop.year_built),
        ]
        + [0.0] * 12
    )
    results = search_comparable(query_vec, top_k=body.top_k)
    return ComparableResponse(comparables=results, query_vector_dim=len(query_vec))


@app.get(
    "/api/v1/neighborhood-stats/{zipcode}",
    response_model=NeighborhoodStatsResponse,
    tags=["Analytics"],
)
async def neighborhood_stats(
    zipcode: str, db: Session = Depends(get_db)
) -> NeighborhoodStatsResponse:
    from app.database import NeighborhoodStat

    stat = db.query(NeighborhoodStat).filter(NeighborhoodStat.zipcode == zipcode).first()
    if stat is None:
        return NeighborhoodStatsResponse(
            zipcode=zipcode,
            median_price=350_000.0,
            median_price_per_sqft=220.0,
            school_score=6.5,
            transit_score=5.5,
            walkability_score=5.5,
            crime_rate=0.3,
            avg_rental_yield=0.065,
        )
    return NeighborhoodStatsResponse(
        zipcode=stat.zipcode,
        median_price=stat.median_price,
        median_price_per_sqft=stat.median_price_per_sqft,
        school_score=stat.school_score,
        transit_score=stat.transit_score,
        walkability_score=stat.walkability_score,
        crime_rate=stat.crime_rate,
        avg_rental_yield=stat.avg_rental_yield,
    )


@app.get("/api/v1/drift-status", response_model=DriftStatusResponse, tags=["Monitoring"])
async def drift_status(db: Session = Depends(get_db)) -> DriftStatusResponse:
    reports = get_drift_summary(db)
    total = len(get_recent_predictions(db, limit=10000))
    return DriftStatusResponse(drift_reports=reports, total_predictions=total)


@app.post("/api/v1/run-drift-check", tags=["Monitoring"])
async def trigger_drift_check(db: Session = Depends(get_db)) -> dict:
    recent = get_recent_predictions(db, limit=200)
    if len(recent) < 20:
        return {"status": "skipped", "reason": "insufficient predictions"}
    vals = [r.predicted_value for r in recent]
    mid = len(vals) // 2
    result = run_drift_check(db, "predicted_value", vals[:mid], vals[mid:])
    return {
        "feature": "predicted_value",
        "ks_statistic": result.ks_statistic,
        "p_value": result.p_value,
        "drift_detected": result.drift_detected,
    }
