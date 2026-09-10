"""Retraining API endpoint logic for Property-Sage.

Separates the retraining route from main.py to keep the main module focused
on request handling. The retrain endpoint is protected by a simple API key
check to prevent accidental or malicious triggers in production.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.monitoring import check_prediction_drift
from pipelines.retrain_dag import get_retrain_history, run_retrain_pipeline

logger = logging.getLogger(__name__)

RETRAIN_API_KEY: str = os.getenv("RETRAIN_API_KEY", "dev-retrain-key")

router = APIRouter(prefix="/api/v1", tags=["Retraining"])


@router.post(
    "/retrain",
    summary="Trigger model retraining",
    response_description="Retraining run result with metrics",
)
async def retrain(
    n_samples: int = 2000,
    force: bool = False,
    x_retrain_key: str | None = Header(default=None),
) -> dict[str, Any]:
    """Trigger an automated model retraining run.

    Requires the ``X-Retrain-Key`` header to match the ``RETRAIN_API_KEY``
    environment variable.

    Args:
        n_samples: Number of synthetic training samples to generate.
        force: If True, retrain even when no drift is detected.
        x_retrain_key: API key header for authorisation.

    Returns:
        Dict describing the retraining run outcome and model metrics.

    Raises:
        HTTPException 401: If the retrain API key is missing or wrong.
    """
    if x_retrain_key != RETRAIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Retrain-Key header",
        )

    drift = [{"drift_detected": force, "p_value": 0.0 if force else 1.0}]
    logger.info("Retrain triggered via API — n_samples=%d force=%s", n_samples, force)
    result = run_retrain_pipeline(n_samples=n_samples, drift_results=drift)
    return result


@router.get(
    "/retrain/history",
    summary="List historical retraining runs",
)
async def retrain_history() -> dict[str, Any]:
    """Return the log of all past retraining runs.

    Returns:
        Dict with a list of run records, newest first.
    """
    return {"history": get_retrain_history()}
