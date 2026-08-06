# Energy-Seer Design

## Problem

Smart meters generate continuous energy readings. Operators need:
1. Short-horizon consumption forecasts for grid balancing
2. Real-time anomaly detection for fault and theft detection
3. Drift alerts when model inputs change over time

## Solution

- **Ensemble ML**: XGBoost + LightGBM + RandomForest soft-voting regressor
- **Feature Engineering**: 7-stage sklearn pipeline with lag, rolling, cyclical temporal, and weather features
- **Anomaly Detection**: IsolationForest + Z-score hybrid for low false-positive rate
- **Drift Monitoring**: KS-test + PSI per feature, triggered on every prediction batch
- **RAG**: FAISS similarity search for nearest historical pattern lookup

## Tradeoffs

- SQLite in dev avoids PostgreSQL setup overhead; Alembic handles schema migrations for both
- Synthetic data at startup means zero cold-start requirement
- IsolationForest contamination=0.05 is tunable for deployment-specific anomaly rates
