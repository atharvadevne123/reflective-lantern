"""Automated retraining pipeline for Forge-Guard.

Follows an Airflow-compatible DAG structure so it can be scheduled
via Airflow or run standalone via ``python retrain_dag.py``.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
MIN_SAMPLES = int(os.getenv("RETRAIN_MIN_SAMPLES", "500"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))


def task_extract_training_data() -> pd.DataFrame:
    """Extract recent prediction logs from the database for retraining."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        db_url = os.getenv("DATABASE_URL", "sqlite:///./forge_guard.db")
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        db = Session()

        from app.database import PredictionLog

        records = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(5000).all()
        db.close()

        if len(records) < MIN_SAMPLES:
            logger.warning("Only %d records — need %d for retraining.", len(records), MIN_SAMPLES)
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {
                    "temperature": r.temperature,
                    "pressure": r.pressure,
                    "vibration": r.vibration,
                    "cycle_time": r.cycle_time,
                    "tool_wear": r.tool_wear,
                    "power_consumption": r.power_consumption,
                    "humidity": r.humidity,
                    "defect": r.prediction,
                }
                for r in records
            ]
        )
        logger.info("Extracted %d records for retraining.", len(df))
        return df
    except Exception:
        logger.exception("Data extraction failed — falling back to synthetic data.")
        from app.features import generate_synthetic_data

        return generate_synthetic_data(n_samples=2000)


def task_check_drift() -> bool:
    """Return True if any feature has drifted enough to warrant retraining."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        db_url = os.getenv("DATABASE_URL", "sqlite:///./forge_guard.db")
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        db = Session()

        from app.database import DriftReport

        recent = db.query(DriftReport).order_by(DriftReport.timestamp.desc()).limit(7).all()
        db.close()

        drifted = [r for r in recent if r.drift_detected]
        if drifted:
            logger.info("Drift detected on features: %s", [r.feature for r in drifted])
            return True
        return False
    except Exception:
        logger.warning("Could not check drift — proceeding with scheduled retrain.")
        return True


def task_engineer_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Apply feature engineering pipeline to the training DataFrame."""
    from app.features import build_feature_pipeline

    feature_cols = [c for c in df.columns if c != "defect"]
    feat_pipe = build_feature_pipeline()
    X = feat_pipe.fit_transform(df[feature_cols])
    y = df["defect"].values
    logger.info("Feature matrix shape: %s", X.shape)
    return X, y


def task_retrain_model(X: np.ndarray, y: np.ndarray) -> dict:
    """Train new model and compare AUC against production model."""
    from app.model import get_metrics, train_model

    old_metrics = get_metrics()
    old_auc = old_metrics.get("auc_cv_mean", 0.0)

    _, new_metrics = train_model(X, y)
    new_auc = new_metrics.get("auc_cv_mean", 0.0)

    logger.info("AUC before: %.4f → after: %.4f", old_auc, new_auc)
    return {"auc_before": old_auc, "auc_after": new_auc, "improved": new_auc >= old_auc - 0.01}


def task_record_run(run_result: dict, trigger: str = "scheduled") -> None:
    """Write retraining result to database."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        db_url = os.getenv("DATABASE_URL", "sqlite:///./forge_guard.db")
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        db = Session()

        from app.database import RetrainingRun

        run = RetrainingRun(
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
            trigger=trigger,
            auc_before=run_result.get("auc_before"),
            auc_after=run_result.get("auc_after"),
            status="success" if run_result.get("improved") else "degraded",
        )
        db.add(run)
        db.commit()
        db.close()
    except Exception:
        logger.exception("Could not record retraining run.")


def run_pipeline(trigger: str = "scheduled") -> None:
    """Execute the full retraining pipeline end-to-end."""
    logger.info("=== Forge-Guard Retraining Pipeline START (trigger=%s) ===", trigger)

    df = task_extract_training_data()
    if df.empty:
        logger.warning("No training data — aborting pipeline.")
        return

    should_retrain = task_check_drift() or trigger == "manual"
    if not should_retrain:
        logger.info("No drift detected and trigger is scheduled — skipping retrain.")
        return

    X, y = task_engineer_features(df)
    result = task_retrain_model(X, y)
    task_record_run(result, trigger=trigger)

    try:
        from app.aws_stub import upload_model

        upload_model()
    except Exception:
        logger.debug("S3 upload unavailable — model kept local only.")

    logger.info(
        "=== Retraining COMPLETE — AUC %.4f → %.4f ===",
        result["auc_before"],
        result["auc_after"],
    )


if __name__ == "__main__":
    trigger = sys.argv[1] if len(sys.argv) > 1 else "manual"
    run_pipeline(trigger=trigger)
