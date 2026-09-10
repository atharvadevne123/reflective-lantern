"""API versioning router — mounts /api/v1/... alongside root endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["v1"])


def include_v1_routes(app_instance) -> None:
    """Attach versioned /api/v1 duplicates of all core routes."""
    from .main import predict_regime, health, metrics, version, drift_check

    router.add_api_route("/predict", predict_regime, methods=["POST"])
    router.add_api_route("/health", health, methods=["GET"])
    router.add_api_route("/metrics", metrics, methods=["GET"])
    router.add_api_route("/version", version, methods=["GET"])
    router.add_api_route("/drift-check", drift_check, methods=["POST"])
    app_instance.include_router(router)
