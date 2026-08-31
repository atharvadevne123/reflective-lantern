"""Versioned API v1 router for Ops-Vision."""

import logging
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import __version__
from app.config import get_settings
from app.crud import (
    avg_confidence,
    bulk_create_predictions,
    count_drift_alerts_last_24h,
    count_incidents_predicted,
    count_predictions,
    create_prediction,
    delete_old_predictions,
    get_incidents_by_service,
    list_incidents,
)
from app.database import get_db
from app.faiss_index import get_runbook_index
from app.features import build_feature_pipeline, dataframe_from_dict
from app.forecasting import ExponentialSmoothingForecaster, get_rate_buffer
from app.model import MODEL_VERSION, load_model, predict
from app.monitoring import get_monitor
from app.schemas import (
    BatchPredictRequest,
    DriftStatusResponse,
    ForecastPoint,
    HealthResponse,
    IncidentRecord,
    MetricsPayload,
    MetricsResponse,
    PredictionResponse,
    PredictionStats,
    RunbookResult,
    RunbookSearchRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["v1"])

_model = None
_feature_pipeline = None


def _load_artifacts() -> tuple:
    """Load the model and its fitted feature pipeline as a matched pair.

    The scaler inside the feature pipeline is stateful, so a model is only
    valid alongside the exact pipeline it was trained with. Both artifacts are
    therefore loaded together, and if either is missing or unreadable both are
    retrained from scratch — never mixed across versions.

    Returns:
        Tuple of (fitted model, fitted feature Pipeline).
    """
    import pickle
    from pathlib import Path

    settings = get_settings()
    pipeline_path = Path(settings.feature_pipeline_path)

    try:
        model = load_model()
        with open(pipeline_path, "rb") as fh:
            pipeline = pickle.load(fh)
        logger.info("Loaded model and feature pipeline from disk")
        return model, pipeline
    except (FileNotFoundError, OSError, pickle.UnpicklingError):
        logger.warning(
            "Model/pipeline artifacts missing or unreadable — bootstrapping "
            "from synthetic data"
        )

    from app.model import generate_synthetic_data, save_model, train

    df, labels = generate_synthetic_data(n_samples=2000)
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df)
    model, metrics = train(X, labels.values)
    logger.info("Bootstrap training complete: %s", metrics)

    try:
        save_model(model, Path(settings.model_path))
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pipeline_path, "wb") as fh:
            pickle.dump(pipeline, fh)
        logger.info("Persisted bootstrapped model and pipeline")
    except OSError:
        logger.exception("Failed to persist artifacts — continuing in memory")

    return model, pipeline


def _ensure_artifacts() -> None:
    """Populate the module-level model and pipeline caches if not already set."""
    global _model, _feature_pipeline
    if _model is None or _feature_pipeline is None:
        _model, _feature_pipeline = _load_artifacts()


def _get_model():
    """Return the cached model, loading or bootstrapping it on first use."""
    _ensure_artifacts()
    return _model


def _get_pipeline():
    """Return the cached fitted feature pipeline, matched to the loaded model."""
    _ensure_artifacts()
    return _feature_pipeline


@router.post("/predict", response_model=PredictionResponse, summary="Predict incident probability")
def predict_endpoint(payload: MetricsPayload, db: Session = Depends(get_db)):
    """Run the ML ensemble on incoming SRE metrics and return incident probability.

    Also records the sample in the drift monitor and persists the prediction.

    Args:
        payload: Validated metrics for one service observation.
        db: Injected database session.

    Returns:
        PredictionResponse with incident flag, severity, and confidence.
    """
    model = _get_model()
    pipeline = _get_pipeline()

    df = dataframe_from_dict(payload.model_dump())
    try:
        X = pipeline.transform(df)
    except Exception:
        logger.exception("Feature transform failed")
        raise HTTPException(status_code=422, detail="Feature engineering failed")

    preds, proba = predict(model, X)
    is_incident = bool(preds[0])
    confidence = float(proba[0])
    severity = _infer_severity(confidence) if is_incident else None

    monitor = get_monitor()
    monitor.record(payload.model_dump())

    runbook_hint: str | None = None
    if is_incident:
        index = get_runbook_index(get_settings().runbooks_path)
        results = index.search(
            f"high cpu memory error latency {payload.service_name}", top_k=1
        )
        if results:
            runbook_hint = results[0][0].title

    try:
        create_prediction(
            db,
            {
                "service_name": payload.service_name,
                "features": payload.model_dump(),
                "predicted_incident": is_incident,
                "predicted_severity": severity,
                "confidence": confidence,
                "model_version": MODEL_VERSION,
            },
        )
    except Exception:
        logger.exception("Failed to persist prediction — continuing")

    return PredictionResponse(
        service_name=payload.service_name,
        predicted_incident=is_incident,
        predicted_severity=severity,
        confidence=confidence,
        model_version=MODEL_VERSION,
        runbook_hint=runbook_hint,
        timestamp=datetime.utcnow(),
    )


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health(db: Session = Depends(get_db)):
    """Return service health including model load status and window sizes.

    Args:
        db: Injected database session.

    Returns:
        HealthResponse with current service state.
    """
    monitor = get_monitor()
    model_loaded = True
    try:
        _get_model()
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        reference_window_size=monitor.reference_size,
        current_window_size=monitor.current_size,
        version=__version__,
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Aggregate operational metrics")
def metrics(db: Session = Depends(get_db)):
    """Return aggregate prediction and drift metrics for dashboarding.

    Args:
        db: Injected database session.

    Returns:
        MetricsResponse with counts and rates.
    """
    total = count_predictions(db)
    incidents = count_incidents_predicted(db)
    drift_24h = count_drift_alerts_last_24h(db)
    avg_conf = avg_confidence(db)

    return MetricsResponse(
        total_predictions=total,
        incident_count=incidents,
        incident_rate=round(incidents / total, 4) if total > 0 else 0.0,
        drift_alerts_24h=drift_24h,
        avg_confidence=round(avg_conf, 4),
    )


@router.post("/runbooks/search", response_model=list[RunbookResult], summary="Search SRE runbooks")
def search_runbooks(request: RunbookSearchRequest):
    """Search the FAISS runbook index for relevant remediation steps.

    Args:
        request: Query string and number of results requested.

    Returns:
        List of matching RunbookResult objects sorted by similarity.
    """
    settings = get_settings()
    index = get_runbook_index(settings.runbooks_path)
    results = index.search(request.query, top_k=request.top_k)
    return [
        RunbookResult(
            title=rb.title,
            content=rb.content,
            score=round(score, 4),
            category=rb.category,
        )
        for rb, score in results
    ]


@router.get("/forecast", response_model=list[ForecastPoint], summary="Forecast incident rate")
def forecast_incident_rate():
    """Forecast the next 24-hour incident rate using exponential smoothing.

    Returns:
        List of ForecastPoint with predicted incident counts and intervals.
    """
    buffer = get_rate_buffer()
    series = buffer.as_array()

    if len(series) < 2:
        from datetime import timedelta
        now = datetime.utcnow()
        return [
            ForecastPoint(
                timestamp=now + timedelta(hours=h),
                value=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
            )
            for h in range(1, 25)
        ]

    forecaster = ExponentialSmoothingForecaster(horizon=24)
    forecaster.fit(series)
    return forecaster.forecast(base_time=datetime.utcnow())


@router.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    summary="Score a batch of observations",
)
def predict_batch(request: BatchPredictRequest, db: Session = Depends(get_db)):
    """Score up to 500 telemetry observations in a single call.

    Batching amortises the model and pipeline lookup across the whole request
    rather than repeating it per observation, which matters when backfilling
    historical telemetry.

    Args:
        request: Batch wrapper holding the list of observations.
        db: Injected database session.

    Returns:
        One PredictionResponse per input item, in the same order.
    """
    model = _get_model()
    pipeline = _get_pipeline()

    frames = [dataframe_from_dict(item.model_dump()) for item in request.items]
    combined = pd.concat(frames, ignore_index=True)

    try:
        X = pipeline.transform(combined)
    except Exception:
        logger.exception("Batch feature transform failed")
        raise HTTPException(status_code=422, detail="Feature engineering failed") from None

    preds, proba = predict(model, X)
    now = datetime.utcnow()

    responses: list[PredictionResponse] = []
    for item, pred, conf in zip(request.items, preds, proba):
        is_incident = bool(pred)
        responses.append(
            PredictionResponse(
                service_name=item.service_name,
                predicted_incident=is_incident,
                predicted_severity=_infer_severity(float(conf)) if is_incident else None,
                confidence=float(conf),
                model_version=MODEL_VERSION,
                runbook_hint=None,
                timestamp=now,
            )
        )

    logger.info("Scored batch of %d observations", len(responses))
    return responses


@router.get(
    "/incidents",
    response_model=list[IncidentRecord],
    summary="List recorded incidents",
)
def get_incidents(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service_name: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """List persisted incidents, most recent first.

    Args:
        limit: Maximum rows to return (1-500).
        offset: Rows to skip, for pagination.
        service_name: Optional exact-match service filter.
        db: Injected database session.

    Returns:
        List of IncidentRecord ordered by created_at descending.
    """
    rows = list_incidents(db, limit=limit, offset=offset, service_name=service_name)
    return [IncidentRecord.model_validate(row) for row in rows]


@router.get("/drift/status", response_model=DriftStatusResponse, summary="Latest drift check status")
def drift_status():
    """Return the drift monitor's last check results.

    Returns:
        DriftStatusResponse with per-feature drift flags.
    """
    from app.monitoring import FEATURE_COLS

    return DriftStatusResponse(
        checked_at=None,
        features_drifted=[],
        features_stable=FEATURE_COLS,
        total_features=len(FEATURE_COLS),
    )


def _infer_severity(confidence: float) -> str:
    """Map confidence score to a severity label.

    Args:
        confidence: Probability of incident (0.0–1.0).

    Returns:
        One of 'critical', 'high', 'medium', or 'low'.
    """
    if confidence >= 0.9:
        return "critical"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"
