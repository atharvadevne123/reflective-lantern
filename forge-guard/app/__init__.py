"""Forge-Guard manufacturing defect prediction application."""

__version__ = "1.0.0"
__author__ = "Reflective Lantern"

from app.database import init_db
from app.features import build_feature_pipeline, engineer_single, generate_synthetic_data
from app.model import load_model, predict, train_model
from app.monitoring import compute_drift, defect_rate, log_prediction, run_drift_check

__all__ = [
    "__version__",
    "build_feature_pipeline",
    "compute_drift",
    "defect_rate",
    "engineer_single",
    "generate_synthetic_data",
    "init_db",
    "load_model",
    "log_prediction",
    "predict",
    "run_drift_check",
    "train_model",
]
