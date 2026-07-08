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
    print(DIAGRAM)
    output_path = "screenshots/architecture.txt"
    try:
        import os
        os.makedirs("screenshots", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(DIAGRAM)
        print(f"Diagram written to {output_path}")
    except Exception as e:
        print(f"Could not write file: {e}")


if __name__ == "__main__":
    main()
