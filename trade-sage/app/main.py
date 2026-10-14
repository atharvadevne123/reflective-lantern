"""FastAPI application for Trade-Sage equity signal prediction service."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy.orm import Session

from app.cache import _prediction_cache
from app.database import get_db, init_db
from app.features import FEATURE_NAMES, engineer_features
from app.middleware import CorrelationIDMiddleware, RateLimitMiddleware
from app.model import load_model, predict, read_champion_auc
from app.monitoring import check_drift_all, log_prediction, record_training_run
from app.portfolio import compute_portfolio_risk
from app.validation import (
    DriftCheckRequest,
    HealthResponse,
    MetricsResponse,
    PortfolioRiskRequest,
    PortfolioRiskResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": %(message)s}',
)
logger = logging.getLogger(__name__)

_model = None
_start_time = time.monotonic()
_prediction_counter = 0
_drift_alert_counter = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    init_db()
    try:
        _model = load_model()
        logger.info('"Trade-Sage model loaded"')
    except FileNotFoundError:
        logger.warning('"No trained model found — /train endpoint required"')
    yield


app = FastAPI(
    title="Trade-Sage",
    description="Equity trading signal prediction and portfolio risk scoring API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CorrelationIDMiddleware)


def _ohlcv_to_df(ohlcv_list) -> pd.DataFrame:
    return pd.DataFrame([row.model_dump() for row in ohlcv_list])


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health_check(db: Session = Depends(get_db)):
    """Service health and readiness check."""
    db_ok = False
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        model_loaded=_model is not None,
        db_connected=db_ok,
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["system"])
def get_metrics():
    """Runtime metrics: prediction count, drift alerts, model AUC."""
    return MetricsResponse(
        total_predictions=_prediction_counter,
        drift_alerts=_drift_alert_counter,
        model_auc=read_champion_auc() or None,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["prediction"])
def predict_signal(req: PredictRequest, db: Session = Depends(get_db)):
    """Predict directional equity signal (BUY/SELL) for given OHLCV history."""
    global _model, _prediction_counter

    cache_key = f"{req.ticker}:{req.horizon_days}:{len(req.ohlcv)}"
    cached = _prediction_cache.get(cache_key)
    if cached:
        return cached

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not trained. POST /api/v1/train first.")

    df = _ohlcv_to_df(req.ohlcv)
    try:
        features = engineer_features(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}") from e

    X = features[-1:, :]
    preds, confs = predict(_model, X)

    prediction = preds[0]
    confidence = confs[0]
    signal = "BUY" if prediction == 1 else "SELL"

    _prediction_counter += 1

    log_prediction(
        db,
        ticker=req.ticker,
        prediction=prediction,
        confidence=confidence,
        horizon_days=req.horizon_days,
        features={"n_rows": len(req.ohlcv)},
    )

    result = PredictResponse(
        ticker=req.ticker,
        prediction=prediction,
        confidence=round(confidence, 4),
        horizon_days=req.horizon_days,
        signal=signal,
    )
    _prediction_cache.set(cache_key, result)
    return result


@app.post("/api/v1/portfolio/risk", response_model=PortfolioRiskResponse, tags=["portfolio"])
def portfolio_risk(req: PortfolioRiskRequest):
    """Compute VaR, Expected Shortfall, volatility, Sharpe, and max drawdown."""
    weights = [h.weight for h in req.holdings]
    price_series = [[row.close for row in h.ohlcv] for h in req.holdings]

    try:
        risk = compute_portfolio_risk(weights, price_series, req.confidence_level)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return PortfolioRiskResponse(**risk)


@app.post("/api/v1/drift", tags=["monitoring"])
def check_drift(req: DriftCheckRequest, db: Session = Depends(get_db)):
    """Check feature drift between reference and current OHLCV window."""
    global _drift_alert_counter
    df = _ohlcv_to_df(req.ohlcv)
    try:
        features = engineer_features(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}") from e

    results = check_drift_all(features, FEATURE_NAMES, db=db)
    n_drift = sum(1 for v in results.values() if v["drift_detected"])
    _drift_alert_counter += n_drift
    return {"drift_results": results, "features_drifted": n_drift, "total_features": len(results)}


@app.post("/api/v1/train", tags=["admin"])
def train_endpoint(db: Session = Depends(get_db)):
    """Trigger a quick synthetic training run (demo — replace with real data in prod)."""
    global _model
    from app.features import FEATURE_NAMES
    from app.model import train_model

    rng = np.random.default_rng(42)
    n = 2000
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    # Synthetic signal: bullish when momentum + RSI features are positive
    score = 0.4 * X[:, 0] + 0.3 * X[:, 7] + 0.3 * rng.standard_normal(n)
    y = (score > 0).astype(int)

    pipe, metrics = train_model(X, y)
    record_training_run(db, metrics)
    _model = pipe
    return {"status": "trained", "metrics": metrics}
