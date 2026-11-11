"""Generate ASCII architecture diagram for Risk-Sense."""

from __future__ import annotations

import os

DIAGRAM = """
Risk-Sense Architecture
=======================

  ┌─────────────────────────────────────────────────┐
  │               Risk-Sense API (FastAPI)           │
  │                                                   │
  │  POST /api/v1/score   GET /api/v1/health         │
  │  GET  /api/v1/metrics GET /api/v1/drift           │
  │  GET  /api/v1/version                             │
  └───────────────┬────────────────────────────────-─┘
                  │
         ┌────────▼────────┐
         │  Feature Pipeline│
         │  (sklearn)        │
         │  LogIncome        │
         │  Interactions     │
         │  ClipOutliers     │
         │  StandardScaler   │
         └────────┬─────────┘
                  │
     ┌────────────▼─────────────┐
     │     Ensemble Model        │
     │  XGBoost + LightGBM + RF  │
     │  CalibratedClassifierCV   │
     │  Voting: soft             │
     └────────────┬─────────────┘
                  │
     ┌────────────▼─────────────┐
     │      Monitoring           │
     │  KS-test drift detection  │
     │  PSI computation          │
     │  Rolling default rate     │
     │  Prediction logging       │
     └────────────┬─────────────┘
                  │
     ┌────────────▼─────────────┐
     │      PostgreSQL           │
     │  credit_applications      │
     │  drift_logs               │
     └───────────────────────────┘
"""


def main() -> None:
    os.makedirs("screenshots", exist_ok=True)
    with open("screenshots/architecture.txt", "w") as f:
        f.write(DIAGRAM)
    print(DIAGRAM)
    print("Architecture diagram written to screenshots/architecture.txt")


if __name__ == "__main__":
    main()
