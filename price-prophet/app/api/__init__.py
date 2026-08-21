"""
API sub-package for Price-Prophet.

Modules
-------
main       - FastAPI application factory and entry point
schemas    - Pydantic request/response models
routes     - versioned API router with pricing endpoints
middleware - custom middleware (timing, logging, etc.)
"""

__all__ = ["main", "middleware", "routes", "schemas"]
