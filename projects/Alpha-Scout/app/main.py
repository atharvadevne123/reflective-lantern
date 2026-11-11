"""FastAPI application with /predict, /health, /metrics, /drift endpoints."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.features import StockFeatureEngineer, make_sample_ohlcv
from app.model import load_model, predict_alpha
from app.monitoring import get_prediction_metrics, log_prediction, run_drift_check
from app.schemas import (
    DriftRequest,
    DriftResponse,
    HealthResponse,
    MetricsResponse,
    OHLCVInput,
    PredictResponse,
)

logger = structlog.get_logger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    init_db()
    _model = load_model()
    logger.info("Alpha-Scout API ready")
    yield
    logger.info("Alpha-Scout API shutting down")


app = FastAPI(
    title="Alpha-Scout API",
    description="ML-powered portfolio alpha generation: score securities on 20+ signals.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.monotonic()
    response: Response = await call_next(request)
    elapsed = round((time.monotonic() - start) * 1000, 1)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    logger.info(
        "request path=%s status=%d elapsed_ms=%s corr_id=%s",
        request.url.path,
        response.status_code,
        elapsed,
        correlation_id,
    )
    return response


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Operations"], summary="Service health check")
async def health(db: Session = Depends(get_db)) -> HealthResponse:
    db_ok = True
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_ok = False
    return HealthResponse(status="ok", model_loaded=_model is not None, db_ok=db_ok)


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Scoring"], summary="Score a security")
async def predict(payload: OHLCVInput, db: Session = Depends(get_db)) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    history = make_sample_ohlcv(60, ticker=payload.ticker)
    history.iloc[-1] = {
        "ticker": payload.ticker,
        "close": payload.close,
        "open": payload.open,
        "high": payload.high,
        "low": payload.low,
        "volume": payload.volume,
    }
    if payload.pe_ratio is not None:
        history["pe_ratio"] = payload.pe_ratio
    if payload.market_cap is not None:
        history["market_cap"] = payload.market_cap
    eng = StockFeatureEngineer()
    eng.fit(history)
    features = eng.transform(history)
    result = predict_alpha(_model, features)
    record = log_prediction(
        db=db,
        ticker=payload.ticker,
        alpha_score=result["alpha_score"],
        return_prediction=result["return_prediction"],
        confidence=result["confidence"],
        model_version=result["model_version"],
    )
    return PredictResponse(
        ticker=payload.ticker,
        alpha_score=result["alpha_score"],
        return_prediction=result["return_prediction"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        prediction_id=record.id,
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Operations"], summary="Prediction aggregate metrics")
async def metrics(db: Session = Depends(get_db)) -> MetricsResponse:
    return MetricsResponse(**get_prediction_metrics(db))


@app.post("/api/v1/drift", response_model=DriftResponse, tags=["Monitoring"], summary="KS-test drift check")
async def drift_check(payload: DriftRequest, db: Session = Depends(get_db)) -> DriftResponse:
    result = run_drift_check(db, payload.feature_name, payload.values)
    return DriftResponse(
        feature_name=payload.feature_name,
        ks_statistic=result["ks_statistic"],
        p_value=result["p_value"],
        drift_detected=result["drift_detected"],
    )


_rate_limit_store: dict[str, list[float]] = {}
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW = 60.0


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    import time as _time
    client_ip = request.client.host if request.client else "unknown"
    now = _time.monotonic()
    hits = _rate_limit_store.setdefault(client_ip, [])
    _rate_limit_store[client_ip] = [t for t in hits if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
    _rate_limit_store[client_ip].append(now)
    return await call_next(request)


class PortfolioRankRequest(BaseModel):
    securities: list[OHLCVInput]


class RankedSecurity(BaseModel):
    ticker: str
    alpha_score: float
    return_prediction: float
    confidence: float
    rank: int


@app.post("/api/v1/rank", response_model=list[RankedSecurity], tags=["Scoring"], summary="Rank a portfolio")
async def rank_portfolio(payload: PortfolioRankRequest, db: Session = Depends(get_db)) -> list[RankedSecurity]:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    results = []
    for sec in payload.securities:
        history = make_sample_ohlcv(60, ticker=sec.ticker)
        history.iloc[-1] = {
            "ticker": sec.ticker, "close": sec.close, "open": sec.open,
            "high": sec.high, "low": sec.low, "volume": sec.volume,
        }
        if sec.pe_ratio is not None:
            history["pe_ratio"] = sec.pe_ratio
        if sec.market_cap is not None:
            history["market_cap"] = sec.market_cap
        eng = StockFeatureEngineer()
        eng.fit(history)
        features = eng.transform(history)
        result = predict_alpha(_model, features)
        results.append(result | {"ticker": sec.ticker})
    results.sort(key=lambda r: r["alpha_score"], reverse=True)
    return [
        RankedSecurity(rank=i + 1, ticker=r["ticker"], alpha_score=r["alpha_score"],
                       return_prediction=r["return_prediction"], confidence=r["confidence"])
        for i, r in enumerate(results)
    ]


@app.get("/api/v1/feature-importance", tags=["Scoring"], summary="Return top feature importances")
async def feature_importance() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    xgb_model = _model.named_steps.get("model")
    if xgb_model is None or not hasattr(xgb_model, "feature_importances_"):
        raise HTTPException(status_code=500, detail="model does not expose feature importances")
    from app.features import FEATURE_COLUMNS
    importances = xgb_model.feature_importances_
    n = min(len(importances), len(FEATURE_COLUMNS))
    pairs = sorted(
        zip(FEATURE_COLUMNS[:n], importances[:n].tolist(), strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    return {"feature_importances": [{"feature": f, "importance": round(v, 6)} for f, v in pairs[:10]]}


@app.post("/api/v1/anomaly", tags=["Monitoring"], summary="Detect anomalous input rows")
async def anomaly_check(payload: OHLCVInput) -> dict:
    from app.anomaly import predict_anomaly
    history = make_sample_ohlcv(100, ticker=payload.ticker)
    history.iloc[-1] = {
        "ticker": payload.ticker, "close": payload.close, "open": payload.open,
        "high": payload.high, "low": payload.low, "volume": payload.volume,
    }
    eng = StockFeatureEngineer()
    eng.fit(history)
    features = eng.transform(history)
    return predict_anomaly(features)


@app.get("/api/v1/pipeline-health", tags=["Operations"], summary="Check ML pipeline health")
async def pipeline_health() -> dict:
    from app.monitoring import check_pipeline_health
    if _model is None:
        return {"healthy": False, "issues": ["no model loaded"]}
    return check_pipeline_health(_model)


@app.get("/api/v1/predictions/export", tags=["Operations"], summary="Export recent predictions as JSON")
async def export_predictions(limit: int = 100, db: Session = Depends(get_db)) -> dict:
    from app.database import PredictionLog
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(min(limit, 1000))
        .all()
    )
    return {
        "count": len(rows),
        "predictions": [
            {
                "id": r.id,
                "ticker": r.ticker,
                "alpha_score": r.alpha_score,
                "return_prediction": r.return_prediction,
                "confidence": r.confidence,
                "model_version": r.model_version,
                "created_at": str(r.created_at),
            }
            for r in rows
        ],
    }
