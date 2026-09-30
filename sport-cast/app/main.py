"""Sport-Cast FastAPI application."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from app.database import PlayerPerformance, get_db, init_db
from app.model import load_model, predict_match, score_player_performance
from app.monitoring import (
    compute_drift,
    get_drift_summary,
    get_prediction_metrics,
    log_drift,
    log_prediction,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')

_model = None
_pipeline = None

_REFERENCE_FEATURES: dict[str, list] = {}
_REQUEST_COUNTS: dict[str, int] = {}

MODEL_VERSION = "1.0.0"
RATE_LIMIT = 200


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        _REQUEST_COUNTS[client] = _REQUEST_COUNTS.get(client, 0) + 1
        if _REQUEST_COUNTS[client] > RATE_LIMIT:
            return Response(content='{"detail":"rate limit exceeded"}', status_code=429, media_type="application/json")
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _pipeline
    init_db()
    _model, _pipeline = load_model()
    logger.info("Sport-Cast startup complete, model loaded")
    yield
    logger.info("Sport-Cast shutdown")


app = FastAPI(
    title="Sport-Cast",
    description="Sports match outcome prediction and player performance scoring API.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=512)
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(RateLimitMiddleware)


class MatchFeatures(BaseModel):
    match_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique match identifier")
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    home_wins_last5: int = Field(ge=0, le=5, description="Home team wins in last 5 matches")
    home_draws_last5: int = Field(ge=0, le=5, description="Home team draws in last 5 matches")
    home_losses_last5: int = Field(ge=0, le=5, description="Home team losses in last 5 matches")
    away_wins_last5: int = Field(ge=0, le=5, description="Away team wins in last 5 matches")
    away_draws_last5: int = Field(ge=0, le=5, description="Away team draws in last 5 matches")
    away_losses_last5: int = Field(ge=0, le=5, description="Away team losses in last 5 matches")
    home_goals_avg: float = Field(ge=0.0, le=10.0, description="Home team average goals scored per match")
    home_goals_conceded_avg: float = Field(ge=0.0, le=10.0, description="Home team average goals conceded per match")
    away_goals_avg: float = Field(ge=0.0, le=10.0, description="Away team average goals scored per match")
    away_goals_conceded_avg: float = Field(ge=0.0, le=10.0, description="Away team average goals conceded per match")
    h2h_home_wins: int = Field(ge=0, default=1, description="Head-to-head home wins")
    h2h_away_wins: int = Field(ge=0, default=1, description="Head-to-head away wins")
    h2h_draws: int = Field(ge=0, default=0, description="Head-to-head draws")
    home_ranking: float = Field(ge=1.0, le=500.0, default=20.0, description="Home team world ranking")
    away_ranking: float = Field(ge=1.0, le=500.0, default=20.0, description="Away team world ranking")
    home_elo: float = Field(ge=100.0, le=2500.0, default=1500.0, description="Home team Elo rating")
    away_elo: float = Field(ge=100.0, le=2500.0, default=1500.0, description="Away team Elo rating")
    home_days_rest: int = Field(ge=0, le=365, default=7, description="Home team days since last match")
    away_days_rest: int = Field(ge=0, le=365, default=7, description="Away team days since last match")
    home_is_home_ground: float = Field(ge=0.0, le=1.0, default=1.0, description="1 if playing on home ground")


class PlayerRequest(BaseModel):
    player_id: str
    player_name: str
    team: str
    sport: str = "football"
    goals_avg: float = Field(ge=0.0, le=5.0, default=0.3)
    assists_avg: float = Field(ge=0.0, le=5.0, default=0.2)
    win_rate: float = Field(ge=0.0, le=1.0, default=0.5)
    minutes_played_ratio: float = Field(ge=0.0, le=1.0, default=0.8)
    injury_days_out: int = Field(ge=0, le=365, default=0)


class DriftCheckRequest(BaseModel):
    feature_name: str
    reference_values: list[float] = Field(min_length=2)
    current_values: list[float] = Field(min_length=2)


@app.get("/api/v1/health", tags=["system"], summary="Health check")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION, "model_loaded": _model is not None}


@app.post("/api/v1/predict", tags=["prediction"], summary="Predict match outcome")
def predict_endpoint(request: MatchFeatures, db: Session = Depends(get_db)) -> dict:
    if _model is None or _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = pd.DataFrame([request.model_dump(exclude={"match_id", "home_team", "away_team"})])
    start = time.perf_counter()
    result = predict_match(features, _model, _pipeline)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    log_prediction(
        db,
        match_id=request.match_id,
        home_team=request.home_team,
        away_team=request.away_team,
        **result,
        model_version=MODEL_VERSION,
    )
    return {
        "match_id": request.match_id,
        "home_team": request.home_team,
        "away_team": request.away_team,
        **result,
        "latency_ms": latency_ms,
    }


@app.post("/api/v1/players/score", tags=["players"], summary="Score player performance")
def score_player_endpoint(request: PlayerRequest, db: Session = Depends(get_db)) -> dict:
    scores = score_player_performance(
        goals_avg=request.goals_avg,
        assists_avg=request.assists_avg,
        win_rate=request.win_rate,
        minutes_played_ratio=request.minutes_played_ratio,
        injury_days_out=request.injury_days_out,
    )
    row = PlayerPerformance(
        player_id=request.player_id,
        player_name=request.player_name,
        team=request.team,
        sport=request.sport,
        **scores,
    )
    db.add(row)
    db.commit()
    return {"player_id": request.player_id, "player_name": request.player_name, "team": request.team, **scores}


@app.get("/api/v1/metrics", tags=["monitoring"], summary="Prediction metrics")
def metrics_endpoint(db: Session = Depends(get_db)) -> dict:
    return {
        "model_version": MODEL_VERSION,
        **get_prediction_metrics(db),
    }


@app.post("/api/v1/drift", tags=["monitoring"], summary="Check feature drift")
def drift_endpoint(request: DriftCheckRequest, db: Session = Depends(get_db)) -> dict:
    result = compute_drift(request.reference_values, request.current_values)
    log_drift(
        db,
        feature_name=request.feature_name,
        ks_statistic=result["ks_statistic"],
        p_value=result["p_value"],
        drift_detected=result["drift_detected"],
    )
    return {"feature": request.feature_name, **result}


@app.get("/api/v1/drift/summary", tags=["monitoring"], summary="Drift history")
def drift_summary_endpoint(db: Session = Depends(get_db)) -> dict:
    return {"drift_records": get_drift_summary(db)}


@app.get("/api/v1/predictions/history", tags=["prediction"], summary="Recent prediction history")
def prediction_history_endpoint(limit: int = 20, db: Session = Depends(get_db)) -> dict:
    from app.database import PredictionLog
    rows = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(limit).all()
    return {
        "predictions": [
            {
                "match_id": r.match_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "predicted_outcome": r.predicted_outcome,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }
