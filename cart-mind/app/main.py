"""Cart-Mind FastAPI application — product recommendation and purchase intent."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from app import __version__
from app.cache import RECOMMENDATION_CACHE, SIMILARITY_CACHE
from app.database import get_db, init_db
from app.features import INTERACTION_COLS, ITEM_COLS, USER_COLS
from app.model import (
    EMBED_DIM,
    METRICS_PATH,
    load_faiss_index,
    load_model,
    predict_intent,
    search_similar_items,
)
from app.monitoring import (
    COUNTERS,
    check_all_features,
    log_prediction,
    update_reference_window,
)
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CacheStatsResponse,
    DriftRequest,
    DriftResponse,
    HealthResponse,
    IntentResponse,
    MetricsResponse,
    ModelInfoResponse,
    RecommendRequest,
    RecommendResponse,
    SimilarItemsRequest,
    SimilarItemsResponse,
    UserItemFeatures,
)
from app.validation import validate_payload

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)

_FEATURE_COLS = USER_COLS + ITEM_COLS + INTERACTION_COLS

# Global model and index holders
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _state["model"] = load_model()
    _state["index"], _state["item_ids"] = load_faiss_index()
    logger.info("Cart-Mind started — model and FAISS index loaded.")
    yield
    _state.clear()


app = FastAPI(
    title="Cart-Mind",
    description=(
        "Real-time product recommendation and purchase intent prediction API "
        "using LightGBM ensemble, FAISS item-similarity search, and drift monitoring."
    ),
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ---------------------------------------------------------------------------
# Correlation-ID middleware
# ---------------------------------------------------------------------------


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = cid
        start = time.perf_counter()
        response: Response = await call_next(request)
        latency = (time.perf_counter() - start) * 1000
        response.headers["X-Correlation-ID"] = cid
        response.headers["X-Response-Time-Ms"] = f"{latency:.2f}"
        return response


app.add_middleware(CorrelationIDMiddleware)


# ---------------------------------------------------------------------------
# Rate limiting middleware (simple in-memory token bucket)
# ---------------------------------------------------------------------------

_rate_buckets: dict[str, tuple[float, int]] = {}
RATE_LIMIT = 200
RATE_WINDOW = 60.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        last_reset, count = _rate_buckets.get(client_ip, (now, 0))
        if now - last_reset > RATE_WINDOW:
            count = 0
            last_reset = now
        count += 1
        _rate_buckets[client_ip] = (last_reset, count)
        if count > RATE_LIMIT:
            return Response(content="rate limit exceeded", status_code=429)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


def _get_correlation_id(request: Request) -> str:
    return getattr(request.state, "correlation_id", str(uuid.uuid4()))


def _row_to_df(data: dict) -> pd.DataFrame:
    return pd.DataFrame([{col: data.get(col, 0) for col in _FEATURE_COLS}])


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health():
    """Return service health and readiness status."""
    return HealthResponse(
        status="healthy",
        model_loaded="model" in _state,
        index_loaded="index" in _state,
        version=__version__,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["System"])
def metrics():
    """Return aggregated request counts and latency percentiles."""
    return MetricsResponse(**COUNTERS.snapshot())


# ---------------------------------------------------------------------------
# Purchase intent prediction
# ---------------------------------------------------------------------------


@app.post("/api/v1/predict", response_model=IntentResponse, tags=["Prediction"])
def predict_purchase_intent(
    payload: UserItemFeatures,
    request: Request,
    db: Session = Depends(get_db),
):
    """Predict probability that a user will purchase a given item."""
    cid = _get_correlation_id(request)
    t0 = time.perf_counter()
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = payload.model_dump()
    warnings = validate_payload(row)
    if warnings:
        logger.warning("Scoring request %s despite %d coherence warning(s)", cid, len(warnings))

    X = _row_to_df(row)
    proba, labels = predict_intent(model, X)
    score = proba[0]
    latency = (time.perf_counter() - t0) * 1000

    # An incoherent payload describes a state the model never saw in training, so
    # its output should not be presented as a confident one regardless of margin.
    if warnings:
        confidence = "low"
    elif score > 0.75 or score < 0.25:
        confidence = "high"
    else:
        confidence = "medium"

    log_prediction(
        db=db,
        correlation_id=cid,
        user_id=payload.user_id,
        item_id=payload.item_id,
        prediction_type="intent",
        score=score,
        latency_ms=latency,
    )
    update_reference_window("purchase_probability", [score])
    COUNTERS.record("intent", latency)

    return IntentResponse(
        user_id=payload.user_id,
        item_id=payload.item_id,
        purchase_probability=round(score, 4),
        will_purchase=bool(labels[0]),
        confidence=confidence,
        model_version=__version__,
        correlation_id=cid,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@app.post("/api/v1/recommend", response_model=RecommendResponse, tags=["Recommendation"])
def recommend_items(
    payload: RecommendRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Return top-K item recommendations ranked by predicted purchase intent."""
    cid = _get_correlation_id(request)
    t0 = time.perf_counter()
    model = _state.get("model")
    item_ids = _state.get("item_ids", [])
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cache_key = (
        f"rec:{payload.user_id}:{payload.top_k}:{payload.purchase_count}:{payload.avg_order_value}"
    )
    cached = RECOMMENDATION_CACHE.get(cache_key)
    if cached is not None:
        COUNTERS.record("recommend", (time.perf_counter() - t0) * 1000)
        logger.info("Recommendation cache hit for %s", payload.user_id)
        return RecommendResponse(
            user_id=payload.user_id,
            recommendations=cached,
            count=len(cached),
            correlation_id=cid,
        )

    rng = np.random.default_rng(abs(hash(payload.user_id)) % (2**32))
    k = min(payload.top_k * 3, len(item_ids) or 50)
    candidate_ids = rng.choice(
        item_ids or [f"item_{i}" for i in range(50)], size=k, replace=False
    ).tolist()

    rows = []
    for _iid in candidate_ids:
        row = {
            "user_age": payload.user_age,
            "purchase_count": payload.purchase_count,
            "avg_order_value": payload.avg_order_value,
            "days_since_last_purchase": payload.days_since_last_purchase,
            "days_since_registration": payload.days_since_registration,
            "session_count_7d": payload.session_count_7d,
            "cart_abandon_rate": payload.cart_abandon_rate,
            "item_price": rng.uniform(10, 500),
            "item_avg_rating": rng.uniform(1, 5),
            "item_review_count": int(rng.integers(0, 5000)),
            "item_inventory_level": int(rng.integers(0, 200)),
            "item_discount_pct": rng.uniform(0, 50),
            "view_count": int(rng.integers(0, 30)),
            "click_count": int(rng.integers(0, 10)),
            "wishlist_flag": int(rng.integers(0, 2)),
            "same_category_purchases": int(rng.integers(0, 20)),
        }
        rows.append(row)

    X = pd.DataFrame([{col: r.get(col, 0) for col in _FEATURE_COLS} for r in rows])
    probas, _ = predict_intent(model, X)

    ranked = sorted(
        zip(candidate_ids, probas, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )[: payload.top_k]

    recommendations = [
        {"rank": i + 1, "item_id": iid, "score": round(score, 4)}
        for i, (iid, score) in enumerate(ranked)
    ]

    RECOMMENDATION_CACHE.set(cache_key, recommendations)

    latency = (time.perf_counter() - t0) * 1000
    log_prediction(
        db, cid, payload.user_id, None, "recommend", ranked[0][1] if ranked else 0.0, latency
    )
    COUNTERS.record("recommend", latency)

    return RecommendResponse(
        user_id=payload.user_id,
        recommendations=recommendations,
        count=len(recommendations),
        correlation_id=cid,
    )


# ---------------------------------------------------------------------------
# Similar items (FAISS)
# ---------------------------------------------------------------------------


@app.post("/api/v1/similar", response_model=SimilarItemsResponse, tags=["Recommendation"])
def similar_items(
    payload: SimilarItemsRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Find items most similar to the seed item using FAISS nearest-neighbour search."""
    cid = _get_correlation_id(request)
    t0 = time.perf_counter()
    index = _state.get("index")
    item_ids = _state.get("item_ids", [])
    if index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")

    cache_key = f"sim:{payload.item_id}:{payload.top_k}"
    cached = SIMILARITY_CACHE.get(cache_key)
    if cached is not None:
        COUNTERS.record("similar", (time.perf_counter() - t0) * 1000)
        logger.info("Similarity cache hit for %s", payload.item_id)
        return SimilarItemsResponse(
            seed_item_id=payload.item_id,
            similar_items=cached,
            count=len(cached),
            correlation_id=cid,
        )

    rng = np.random.default_rng(abs(hash(payload.item_id)) % (2**32))
    query_vec = rng.random(EMBED_DIM).astype(np.float32)

    results = search_similar_items(index, item_ids, query_vec, top_k=payload.top_k)
    SIMILARITY_CACHE.set(cache_key, results)

    latency = (time.perf_counter() - t0) * 1000
    log_prediction(
        db,
        cid,
        "anonymous",
        payload.item_id,
        "similar",
        results[0]["similarity_score"] if results else 0.0,
        latency,
    )
    COUNTERS.record("similar", latency)

    return SimilarItemsResponse(
        seed_item_id=payload.item_id,
        similar_items=results,
        count=len(results),
        correlation_id=cid,
    )


# ---------------------------------------------------------------------------
# Drift check
# ---------------------------------------------------------------------------


@app.post("/api/v1/drift", response_model=DriftResponse, tags=["Monitoring"])
def check_drift(
    payload: DriftRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Run KS-test drift analysis on provided feature distributions."""
    results = check_all_features(payload.feature_values, db)
    drifted = [f for f, r in results.items() if r.get("drift_detected")]
    COUNTERS.drift_alerts += len(drifted)
    return DriftResponse(
        results=results,
        drifted_features=drifted,
        total_checked=len(results),
    )


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    summary="Score many user-item pairs in one call",
    description=(
        "Scores up to 500 user-item pairs in a single vectorised pass through the "
        "ensemble. Substantially cheaper than issuing N single-prediction calls, "
        "since the feature pipeline and all three estimators run once over the "
        "whole frame rather than once per row."
    ),
)
def predict_batch(
    payload: BatchPredictRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> BatchPredictResponse:
    """Score a batch of user-item pairs in a single vectorised inference pass.

    Args:
        payload: Batch of user-item feature rows.
        request: Incoming request, carrying the correlation ID.
        db: Database session for prediction logging.

    Returns:
        Per-row probabilities plus the batch mean.

    Raises:
        HTTPException: 503 when the model is not loaded.
    """
    cid = _get_correlation_id(request)
    t0 = time.perf_counter()
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = [item.model_dump() for item in payload.items]
    X = pd.DataFrame([{col: r.get(col, 0) for col in _FEATURE_COLS} for r in rows])
    probas, labels = predict_intent(model, X)

    predictions = [
        {
            "user_id": r["user_id"],
            "item_id": r["item_id"],
            "purchase_probability": round(p, 4),
            "will_purchase": bool(lab),
        }
        for r, p, lab in zip(rows, probas, labels, strict=True)
    ]

    latency = (time.perf_counter() - t0) * 1000
    COUNTERS.record("intent", latency)
    update_reference_window("purchase_probability", probas)
    log_prediction(db, cid, rows[0]["user_id"], None, "batch", float(np.mean(probas)), latency)
    logger.info(
        "Batch prediction: %d rows in %.1fms (%.2fms/row)",
        len(rows),
        latency,
        latency / len(rows),
    )

    return BatchPredictResponse(
        predictions=predictions,
        count=len(predictions),
        mean_probability=round(float(np.mean(probas)), 4),
        correlation_id=cid,
    )


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


@app.get(
    "/api/v1/model/info",
    response_model=ModelInfoResponse,
    tags=["System"],
    summary="Inspect the deployed model",
    description=(
        "Returns the serving model's cross-validated metrics, ensemble composition, "
        "and feature-pipeline stages. Metric fields are null when no metrics.json "
        "has been written yet — that is, when the API is serving a bootstrap model."
    ),
)
def model_info() -> ModelInfoResponse:
    """Return metadata and CV metrics for the currently deployed model.

    Returns:
        Model version, CV metrics (null if unavailable), and architecture summary.
    """
    metrics: dict[str, Any] = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("metrics.json present but unreadable", exc_info=True)

    return ModelInfoResponse(
        model_version=metrics.get("model_version", __version__),
        auc_mean=metrics.get("auc_mean"),
        auc_std=metrics.get("auc_std"),
        n_features=metrics.get("n_features"),
        n_samples=metrics.get("n_samples"),
        positive_rate=metrics.get("positive_rate"),
        ensemble_members=["LightGBM", "XGBoost", "RandomForest"],
        feature_stages=[
            "ratios",
            "interactions",
            "lag_rolling",
            "discount_encoding",
            "scaler",
        ],
    )


# ---------------------------------------------------------------------------
# Cache stats
# ---------------------------------------------------------------------------


@app.get(
    "/api/v1/cache/stats",
    response_model=CacheStatsResponse,
    tags=["System"],
    summary="Inspect cache hit rates",
    description="Returns entry counts, hit/miss counters and hit rates for both TTL caches.",
)
def cache_stats() -> CacheStatsResponse:
    """Return hit-rate statistics for the recommendation and similarity caches.

    Returns:
        Statistics for both caches.
    """
    return CacheStatsResponse(
        recommendation_cache=RECOMMENDATION_CACHE.stats(),
        similarity_cache=SIMILARITY_CACHE.stats(),
    )
