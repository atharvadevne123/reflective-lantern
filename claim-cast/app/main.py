"""FastAPI application for Claim-Cast insurance prediction API."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi import Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.middleware import CorrelationIDMiddleware, RateLimitMiddleware
from app.model import METRICS_PATH, SEVERITY_LABELS, load_model, predict
from app.monitoring import compute_drift, get_recent_scores, log_drift, log_prediction
from app.schemas import (
    ClaimRequest,
    ClaimResponse,
    DriftResponse,
    HealthResponse,
    MetricsResponse,
    PremiumRequest,
    PremiumResponse,
)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger(__name__)

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
_model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_loaded
    init_db()
    try:
        load_model()
        _model_loaded = True
        logger.info("model_loaded", version=MODEL_VERSION)
    except Exception as exc:
        logger.error("model_load_failed", error=str(exc))
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Claim-Cast",
    description="Insurance claim severity prediction and triage scoring API.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Operations"])
def health(db: Session = Depends(get_db)):
    """Liveness and readiness probe."""
    db_ok = True
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_ok = False
    return HealthResponse(
        status="ok", model_loaded=_model_loaded, db_connected=db_ok, version=MODEL_VERSION
    )


@app.post("/api/v1/predict", response_model=ClaimResponse, tags=["Prediction"])
def predict_claim(request: ClaimRequest, db: Session = Depends(get_db)):
    """Predict claim severity and triage priority for a single insurance claim."""
    import pandas as pd

    if not _model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model_not_loaded"
        )

    try:
        X = pd.DataFrame([request.model_dump(exclude={"claim_id"})])
        result = predict(X)
    except Exception as exc:
        logger.error("predict_failed", claim_id=request.claim_id, error=str(exc))
        raise HTTPException(status_code=500, detail="prediction_failed") from exc

    log_prediction(
        db,
        claim_id=request.claim_id,
        severity_score=result["severity_score"],
        severity_label=result["severity_label"],
        triage_priority=result["triage_priority"],
        model_version=MODEL_VERSION,
    )

    logger.info(
        "predict_ok",
        claim_id=request.claim_id,
        severity=result["severity_label"],
        priority=result["triage_priority"],
    )
    return ClaimResponse(claim_id=request.claim_id, model_version=MODEL_VERSION, **result)


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Operations"])
def metrics(db: Session = Depends(get_db)):
    """Return model performance metrics and prediction statistics."""
    from app.database import ClaimPrediction, DriftLog

    raw: dict = {}
    if METRICS_PATH.exists():
        raw = json.loads(METRICS_PATH.read_text())

    total = db.query(ClaimPrediction).count()
    drift_alerts = db.query(DriftLog).filter(DriftLog.drift_detected == 1).count()

    return MetricsResponse(
        f1_macro_mean=raw.get("f1_macro_mean"),
        f1_macro_std=raw.get("f1_macro_std"),
        n_features=raw.get("n_features"),
        n_samples=raw.get("n_samples"),
        severity_classes=raw.get("severity_classes", list(SEVERITY_LABELS.values())),
        total_predictions=total,
        drift_alerts=drift_alerts,
    )


@app.get("/api/v1/drift", response_model=DriftResponse, tags=["Monitoring"])
def drift_check(db: Session = Depends(get_db)):
    """Run KS-test drift detection on recent vs reference severity scores."""

    recent = get_recent_scores(db, limit=100)
    rng = __import__("numpy").random.default_rng(0)
    reference = rng.beta(2, 5, 200).tolist()

    result = compute_drift(reference, recent)

    if result.get("ks_statistic") is not None and not result.get("error"):
        log_drift(
            db,
            feature_name="severity_score",
            ks_statistic=result["ks_statistic"],
            p_value=result["p_value"],
            drift_detected=result["drift_detected"],
            sample_size=len(recent),
        )

    return DriftResponse(
        feature="severity_score",
        ks_statistic=result.get("ks_statistic"),
        p_value=result.get("p_value"),
        drift_detected=result.get("drift_detected", False),
        reference_size=len(reference),
        current_size=len(recent),
    )


@app.post("/api/v1/premium", response_model=PremiumResponse, tags=["Actuarial"])
def adjust_premium(request: PremiumRequest):
    """Calculate risk-adjusted premium recommendation based on claim triage."""
    multipliers = {1: 1.5, 2: 1.25, 3: 1.1, 4: 1.0}
    label_bump = {"critical": 0.2, "high": 0.1, "medium": 0.05, "low": 0.0}

    multiplier = multipliers.get(request.triage_priority, 1.0)
    multiplier += label_bump.get(request.severity_label, 0.0)
    multiplier += min(request.claim_frequency * 0.05, 0.3)

    adjusted = round(request.base_premium * multiplier, 2)
    if multiplier >= 1.4:
        rec = "Flag for underwriting review — significant risk increase detected."
    elif multiplier >= 1.2:
        rec = "Moderate premium adjustment recommended."
    else:
        rec = "Standard renewal pricing applies."

    return PremiumResponse(
        base_premium=request.base_premium,
        risk_multiplier=round(multiplier, 4),
        adjusted_premium=adjusted,
        recommendation=rec,
    )


@app.get("/api/v1/triage/{triage_priority}", tags=["Prediction"])
def triage_summary(triage_priority: int = FPath(..., ge=1, le=4)):
    """Return description and SLA for a triage priority level."""
    slas = {
        1: {
            "level": "P1 — Critical",
            "sla_hours": 4,
            "action": "Immediate senior adjuster assignment",
        },
        2: {"level": "P2 — High", "sla_hours": 24, "action": "Same-day adjuster assignment"},
        3: {"level": "P3 — Medium", "sla_hours": 72, "action": "Standard adjuster queue"},
        4: {"level": "P4 — Low", "sla_hours": 168, "action": "Automated fast-track processing"},
    }
    return slas[triage_priority]


@app.get("/api/v1/claims/stats", tags=["Prediction"])
def claim_stats(db: Session = Depends(get_db)):
    """Return aggregate statistics for all predictions in the database.

    Returns:
        Dict with count breakdowns by severity label and triage priority.
    """
    from sqlalchemy import func

    from app.database import ClaimPrediction

    rows = (
        db.query(ClaimPrediction.severity_label, ClaimPrediction.triage_priority, func.count())
        .group_by(ClaimPrediction.severity_label, ClaimPrediction.triage_priority)
        .all()
    )
    return {
        "breakdown": [
            {"severity_label": r[0], "triage_priority": r[1], "count": r[2]} for r in rows
        ],
        "total": sum(r[2] for r in rows),
    }


@app.post("/api/v1/predict/batch", tags=["Prediction"])
def predict_batch(requests_list: list[ClaimRequest], db: Session = Depends(get_db)):
    """Predict severity for a batch of insurance claims (max 50 per call).

    Args:
        requests_list: List of ClaimRequest objects.
        db: Database session dependency.

    Returns:
        List of ClaimResponse objects.
    """
    import pandas as pd

    if len(requests_list) > 50:
        raise HTTPException(status_code=400, detail="batch_size_exceeds_limit_50")
    if not _model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model_not_loaded"
        )

    results = []
    for req in requests_list:
        try:
            X = pd.DataFrame([req.model_dump(exclude={"claim_id"})])
            result = predict(X)
            log_prediction(
                db,
                req.claim_id,
                result["severity_score"],
                result["severity_label"],
                result["triage_priority"],
                MODEL_VERSION,
            )
            results.append(
                ClaimResponse(claim_id=req.claim_id, model_version=MODEL_VERSION, **result)
            )
        except Exception as exc:
            logger.error("batch_predict_failed", claim_id=req.claim_id, error=str(exc))
            raise HTTPException(
                status_code=500, detail=f"prediction_failed for {req.claim_id}"
            ) from exc
    return results


@app.get("/api/v1/feature-importance", tags=["Prediction"])
def feature_importance():
    """Return feature importance scores from the trained XGBoost model.

    Returns:
        Dict mapping feature names to importance scores, sorted descending.
    """
    from app.features import get_feature_names

    pipe, _ = load_model()
    xgb_model = pipe.named_steps["model"]
    scores = xgb_model.feature_importances_.tolist()
    names = get_feature_names()
    pairs = sorted(zip(names, scores, strict=False), key=lambda x: x[1], reverse=True)
    return {"feature_importance": [{"feature": n, "importance": round(s, 4)} for n, s in pairs]}


@app.post("/api/v1/retrain", tags=["Operations"])
def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining in a background task.

    Returns:
        Dict with status and task id.
    """
    import uuid

    task_id = str(uuid.uuid4())[:8]

    def _run_retrain():
        import sys

        sys.path.insert(0, ".")
        from pipelines.retrain_dag import (
            evaluate_model,
            fetch_training_data,
            promote_model,
            run_retraining,
        )

        path = fetch_training_data()
        metrics = run_retraining(path)
        if evaluate_model(metrics):
            promote_model()

    background_tasks.add_task(_run_retrain)
    return {"status": "retraining_scheduled", "task_id": task_id}
