"""Generate an ASCII architecture diagram for Temporal-Pulse."""

from __future__ import annotations

DIAGRAM = """
Temporal-Pulse — Architecture Diagram
======================================

  ┌─────────────────────────────────────────────────────────┐
  │                    Client / IoT Sensors                  │
  └─────────────────────┬───────────────────────────────────┘
                        │ HTTP POST /api/v1/detect
                        ▼
  ┌─────────────────────────────────────────────────────────┐
  │               FastAPI Application (uvicorn)              │
  │                                                         │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
  │  │  /detect     │  │  /forecast   │  │  /drift      │  │
  │  │  /train      │  │  /health     │  │  /metrics    │  │
  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
  │         │                 │                 │           │
  │  ┌──────▼───────────────────────────────────▼───────┐   │
  │  │           Feature Engineering Pipeline            │   │
  │  │  Rolling Stats • Lags • ROC • Time Encoding      │   │
  │  │  Cross-Sensor Correlation • RobustScaler         │   │
  │  └──────┬───────────────────────────────────────────┘   │
  │         │                                               │
  │  ┌──────▼───────────────────────────────────────────┐   │
  │  │              ML Ensemble                         │   │
  │  │  Isolation Forest (anomaly scoring)              │   │
  │  │  Random Forest   (multi-step forecasting)        │   │
  │  │  FAISS           (nearest-neighbour explanation) │   │
  │  └──────┬───────────────────────────────────────────┘   │
  │         │                                               │
  │  ┌──────▼───────────────────────────────────────────┐   │
  │  │           Monitoring & Drift Detection            │   │
  │  │   KS-test per feature • Latency tracking         │   │
  │  │   Prediction logging  • Anomaly event store      │   │
  │  └──────────────────────────────────────────────────┘   │
  └──────────────────────────┬──────────────────────────────┘
                             │ SQLAlchemy ORM
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │                   PostgreSQL Database                    │
  │  sensor_readings • anomaly_events • predictions         │
  │  drift_logs                                             │
  └─────────────────────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Retraining DAG  │
                    │ (Airflow/cron)  │
                    │ Daily @midnight │
                    └─────────────────┘
"""


def main() -> None:
    """Print the architecture diagram."""
    import logging
    import os

    logger = logging.getLogger(__name__)
    print(DIAGRAM)
    output_path = "screenshots/architecture.txt"
    try:
        os.makedirs("screenshots", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(DIAGRAM)
        logger.info("Diagram written to %s", output_path)
    except Exception as e:
        logger.error("Could not write file: %s", e)


if __name__ == "__main__":
    main()
