"""FastAPI application for Property-Sage valuation API."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.features import NEIGHBORHOODS, PROPERTY_TYPES, property_to_dataframe
from app.model import get_metrics, load_models, predict
from app.monitoring import check_prediction_drift, get_prediction_stats, log_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_price_model = None
_rental_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _price_model, _rental_model
    init_db()
    _price_model, _rental_model = load_models()
    logger.info("Models loaded and ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Property-Sage API",
    description="Real estate property valuation and rental yield prediction API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    logger.info(
        "method=%s path=%s status=%d duration_ms=%s correlation_id=%s",
        request.method, request.url.path, response.status_code, duration_ms, correlation_id,
    )
    return response


class PropertyRequest(BaseModel):
    bedrooms: int = Field(..., ge=1, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0.5, le=15.0, description="Number of bathrooms")
    sqft: float = Field(..., ge=100, le=50_000, description="Square footage")
    lot_size: float | None = Field(None, ge=0, description="Lot size in sqft")
    year_built: int = Field(..., ge=1800, le=2024, description="Year the property was built")
    neighborhood: str = Field(..., description="Neighborhood name")
    property_type: str = Field(..., description="Type of property")

    @field_validator("neighborhood")
    @classmethod
    def validate_neighborhood(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in NEIGHBORHOODS:
            raise ValueError(f"neighborhood must be one of {NEIGHBORHOODS}")
        return v

    @field_validator("property_type")
    @classmethod
    def validate_property_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in PROPERTY_TYPES:
            raise ValueError(f"property_type must be one of {PROPERTY_TYPES}")
        return v


class PredictionResponse(BaseModel):
    request_id: str
    predicted_price: float
    predicted_rental_yield: float
    estimated_annual_rental: float
    estimated_monthly_rental: float
    neighborhood: str
    property_type: str


@app.get("/api/v1/health", tags=["System"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "property-sage", "version": "1.0.0"}


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_endpoint(
    body: PropertyRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    if _price_model is None or _rental_model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models not loaded")

    request_id = str(uuid.uuid4())
    X = property_to_dataframe(body.model_dump())
    output = predict(_price_model, _rental_model, X)

    log_prediction(db, request_id, body.model_dump(), output)

    return PredictionResponse(
        request_id=request_id,
        neighborhood=body.neighborhood,
        property_type=body.property_type,
        **output,
    )


@app.get("/api/v1/metrics", tags=["Monitoring"])
async def metrics_endpoint(db: Session = Depends(get_db)) -> dict[str, Any]:
    model_metrics = get_metrics()
    pred_stats = get_prediction_stats(db)
    return {"model_performance": model_metrics, "prediction_stats": pred_stats}


@app.post("/api/v1/drift-check", tags=["Monitoring"])
async def drift_check(db: Session = Depends(get_db)) -> dict[str, Any]:
    return check_prediction_drift(db)


@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    return {"message": "Property-Sage API — see /docs for usage"}
