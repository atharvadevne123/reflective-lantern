"""
FastAPI application factory for Price-Prophet.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="ML-powered dynamic pricing optimisation for e-commerce",
        version="1.0.0",
        debug=settings.debug,
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
