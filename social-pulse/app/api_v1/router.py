"""Versioned /api/v1 router for Social-Pulse endpoints."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.main import MODEL_VERSION, _model, _pipeline
from app.model import PredictResult, predict
from app.monitoring import log_prediction
from app.schemas import BatchPredictRequest, BatchPredictResponse, PostFeaturesFull

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])


@router.post(
    "/predict",
    summary="Predict virality (versioned)",
    description="Versioned viral content prediction endpoint at /api/v1/predict.",
    response_model=dict,
    status_code=status.HTTP_200_OK,
)
def v1_predict(
    body: PostFeaturesFull,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Versioned single-post prediction endpoint.

    Args:
        body: Validated post features.
        request: Incoming HTTP request (for correlation ID).
        db: Injected database session.

    Returns:
        Dict with viral probability, label, and metadata.

    Raises:
        HTTPException: 503 if model is not loaded, 500 on inference error.
    """
    if _model is None or _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available",
        )

    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    sample = body.model_dump()

    try:
        result: PredictResult = predict(sample, _pipeline, _model, MODEL_VERSION)
    except Exception as exc:
        logger.exception("v1/predict failed for %s", request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        log_prediction(
            db=db,
            request_id=request_id,
            sample=sample,
            viral_probability=result.viral_probability,
            viral_label=result.viral_label,
            ensemble_confidence=result.ensemble_confidence,
            model_version=result.model_version,
            latency_ms=result.latency_ms,
        )
    except Exception:
        logger.warning("DB log failed for %s", request_id)

    return {
        "request_id": request_id,
        "viral_probability": result.viral_probability,
        "viral_label": result.viral_label,
        "ensemble_confidence": result.ensemble_confidence,
        "model_version": result.model_version,
        "latency_ms": result.latency_ms,
        "api_version": "v1",
    }


@router.post(
    "/predict/batch",
    summary="Batch predict virality",
    description="Predict virality for up to 50 posts in a single request.",
    response_model=BatchPredictResponse,
    status_code=status.HTTP_200_OK,
)
def v1_batch_predict(
    body: BatchPredictRequest,
    db: Session = Depends(get_db),
) -> BatchPredictResponse:
    """Batch prediction endpoint — up to 50 posts per call.

    Args:
        body: Validated batch request with list of posts.
        db: Injected database session.

    Returns:
        BatchPredictResponse with per-post results and metadata.
    """
    if _model is None or _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available",
        )

    results: list[dict[str, Any]] = []
    for post in body.posts:
        sample = post.model_dump()
        request_id = str(uuid.uuid4())
        try:
            r = predict(sample, _pipeline, _model, MODEL_VERSION)
            results.append(
                {
                    "request_id": request_id,
                    "viral_probability": r.viral_probability,
                    "viral_label": r.viral_label,
                    "ensemble_confidence": r.ensemble_confidence,
                }
            )
        except Exception as exc:
            logger.error("Batch predict failed for sample: %s", exc)
            results.append({"request_id": request_id, "error": str(exc)})

    return BatchPredictResponse(
        results=results,
        total=len(results),
        model_version=MODEL_VERSION,
    )
