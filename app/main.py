"""FastAPI application — Watt-Guard energy forecasting API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from sqlalchemy.orm import Session

from app import __version__
from app.database import create_tables, get_db
from app.features import make_feature_row
from app.middleware import CorrelationIDMiddleware, RateLimitMiddleware
from app.model import (
    get_feature_importance,
    get_metrics,
    load_anomaly_model,
    load_model,
    predict,
    score_anomaly,
)
from app.monitoring import (
    LatencyTimer,
    compute_drift,
    get_anomaly_stats,
    get_drift_summary,
    get_prediction_stats,
    get_recent_predictions,
    log_anomaly,
    log_prediction,
    run_drift_check,
    set_reference_window,
)
from app.reporting import energy_efficiency_grade, estimate_savings
from app.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    AnomalyStatsResponse,
    ComparableRequest,
    ComparableResponse,
    DriftRequest,
    DriftResponse,
    DriftStatusResponse,
    EnergyReadingIn,
    FeatureImportanceItem,
    FeatureImportanceResponse,
    HealthResponse,
    MetricsResponse,
    NeighborhoodStatsResponse,
    PredictResponse,
    SavingsRequest,
    SavingsResponse,
)
from app.similarity import search_comparable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_model_bundle: dict[str, Any] | None = None
_anomaly_bundle: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
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


@app.get("/health", response_model=HealthResponse, tags=["System"])
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


@app.get("/api/v1/version", tags=["System"])
def version() -> dict[str, str]:
    """Return the current API version string."""
    return {"version": __version__}


@app.get("/api/v1/regions", tags=["System"])
def list_grid_regions() -> list[dict[str, Any]]:
    """List all known grid regions with metadata."""
    from app.regions import list_regions

    return list_regions()


@app.post("/api/v1/predict/batch", tags=["Forecasting"])
def predict_batch(
    payloads: list[EnergyReadingIn],
    db: Session = Depends(get_db),
) -> list[PredictResponse]:
    """Forecast energy consumption for multiple buildings in one request."""
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Forecasting model not loaded.")
    if len(payloads) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds 100.")
    results = []
    for payload in payloads:
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
        results.append(
            PredictResponse(
                building_id=payload.building_id,
                timestamp=payload.timestamp,
                predicted_kwh=round(kwh_pred, 3),
                model_version=__version__,
                latency_ms=timer.ms,
            )
        )
    return results


@app.post("/api/v1/comparable-properties", response_model=ComparableResponse, tags=["Search"])
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
async def neighborhood_stats(zipcode: str, db: Session = Depends(get_db)) -> NeighborhoodStatsResponse:
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
        "status": "completed",
        "feature": "predicted_value",
        "ks_statistic": result.ks_statistic,
        "p_value": result.p_value,
        "drift_detected": result.drift_detected,
    }


@app.get("/api/v1/feature-importance", response_model=FeatureImportanceResponse, tags=["Forecasting"])
def feature_importance(top_n: int = 20) -> FeatureImportanceResponse:
    """Return the top N most important features from the trained forecasting model."""
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run /api/v1/train first.")
    items = get_feature_importance(_model_bundle, top_n=top_n)
    return FeatureImportanceResponse(
        features=[FeatureImportanceItem(**i) for i in items],
        top_n=top_n,
        model_version=__version__,
    )


@app.get("/api/v1/anomaly/stats", response_model=AnomalyStatsResponse, tags=["Anomaly Detection"])
def anomaly_stats(db: Session = Depends(get_db)) -> AnomalyStatsResponse:
    """Return aggregated anomaly detection statistics."""
    stats = get_anomaly_stats(db)
    return AnomalyStatsResponse(**stats)


@app.post("/api/v1/savings", response_model=SavingsResponse, tags=["Reporting"])
def savings_estimate(body: SavingsRequest) -> SavingsResponse:
    """Estimate energy savings and cost reduction versus a baseline consumption series."""
    try:
        result = estimate_savings(body.actual_kwh, body.baseline_kwh, body.tariff_per_kwh)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SavingsResponse(**result)


@app.get("/api/v1/efficiency-grade", tags=["Reporting"])
def efficiency_grade(actual_kwh: float, baseline_kwh: float) -> dict[str, str]:
    """Return an efficiency letter grade (A+…F) for a consumption vs baseline comparison."""
    grade = energy_efficiency_grade(actual_kwh, baseline_kwh)
    return {"grade": grade, "actual_kwh": str(actual_kwh), "baseline_kwh": str(baseline_kwh)}


@app.get("/api/v1/ready", tags=["System"])
def readiness_probe() -> dict[str, str]:
    """Readiness probe: returns 503 until both models are loaded."""
    if _model_bundle is None or _anomaly_bundle is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded.")
    return {"status": "ready"}


@app.post("/api/v1/validate-batch", tags=["Validation"])
def validate_batch_readings(body: dict) -> dict:
    """Validate a batch of energy readings and return per-row error reports."""
    from app.validation import batch_validate_readings

    readings = body.get("readings", [])
    if not readings:
        raise HTTPException(status_code=400, detail="readings list must not be empty")
    results = batch_validate_readings(readings)
    valid_count = sum(1 for r in results if r["valid"])
    return {
        "total": len(results),
        "valid_count": valid_count,
        "invalid_count": len(results) - valid_count,
        "results": results,
    }
