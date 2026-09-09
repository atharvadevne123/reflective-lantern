"""FastAPI application for Signal-Forge market regime detection API."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from .database import create_tables, get_db
from .features import FEATURE_COLS, extract_single_row
from .model import predict
from .monitoring import get_drift_summary, log_prediction, monitor_all_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Signal-Forge starting up")
    create_tables()
    yield
    logger.info("Signal-Forge shutting down")


app = FastAPI(
    title="Signal-Forge",
    description="Real-time financial market regime detection and portfolio stress testing API",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_request_count = 0
_start_time = time.time()


@app.middleware("http")
async def count_requests(request: Request, call_next):
    global _request_count
    _request_count += 1
    response = await call_next(request)
    return response


class PredictRequest(BaseModel):
    """Input schema for /predict endpoint."""

    ticker: str = Field(..., min_length=1, max_length=10, description="Asset ticker symbol")
    close: float = Field(..., gt=0, description="Latest closing price")
    volume: float = Field(..., ge=0, description="Latest traded volume")
    market_return: float = Field(0.0, description="Benchmark daily return (optional)")

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v: str) -> str:
        return v.upper().strip()


class PredictResponse(BaseModel):
    """Output schema for /predict endpoint."""

    ticker: str
    regime: str
    confidence: float
    risk_score: float
    volatility: float
    momentum: float
    correlation: float
    volume_ratio: float
    beta: float
    similar_periods: list[dict[str, Any]]
    model_version: str


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict market regime",
    description="Returns regime classification (bull/bear/sideways/volatile) and portfolio risk score.",
)
def predict_regime(
    req: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Classify market regime for a given asset using ensemble ML model."""
    try:
        import pandas as pd
        row = {"close": req.close, "volume": req.volume, "market_return": req.market_return}
        df = extract_single_row(row)
        from .features import build_feature_pipeline
        pipeline = build_feature_pipeline()
        arr = pipeline.fit_transform(df)
        last_row = arr[-1:]

        raw_values: dict[str, float] = {}
        for i, col in enumerate(FEATURE_COLS):
            raw_values[col] = float(arr[-1, i]) if i < arr.shape[1] else 0.0

        result = predict(last_row, req.ticker, raw_values)
        try:
            log_prediction(
                db,
                ticker=result.ticker,
                regime=result.regime,
                confidence=result.confidence,
                risk_score=result.risk_score,
                features=raw_values,
            )
        except Exception as e:
            logger.warning("Prediction logging failed (non-fatal): %s", e)

        return PredictResponse(
            ticker=result.ticker,
            regime=result.regime,
            confidence=result.confidence,
            risk_score=result.risk_score,
            volatility=result.volatility,
            momentum=result.momentum,
            correlation=result.correlation,
            volume_ratio=result.volume_ratio,
            beta=result.beta,
            similar_periods=result.similar_periods,
            model_version=APP_VERSION,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction error")


@app.get(
    "/health",
    summary="Health check",
    description="Returns service health status.",
)
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok", "version": APP_VERSION}


@app.get(
    "/metrics",
    summary="Service metrics",
    description="Returns request count, uptime, and drift summary.",
)
def metrics(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return runtime metrics and drift detection summary."""
    uptime = round(time.time() - _start_time, 1)
    drift = get_drift_summary(db)
    return {
        "request_count": _request_count,
        "uptime_seconds": uptime,
        "version": APP_VERSION,
        "drift": drift,
    }


@app.get(
    "/version",
    summary="API version",
    description="Returns the current API version.",
)
def version() -> dict[str, str]:
    """Return the API version."""
    return {"version": APP_VERSION}


@app.post(
    "/drift-check",
    summary="Run drift detection",
    description="Execute KS-test drift detection across all tracked features.",
)
def drift_check(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Trigger drift detection and return per-feature results."""
    results = monitor_all_features(db)
    any_drift = any(r.get("drift_detected") for r in results)
    return {"drift_detected": any_drift, "features": results}
