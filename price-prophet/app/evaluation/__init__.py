"""
Evaluation sub-package for Price-Prophet.

Modules
-------
metrics     - regression and revenue metrics (MAE, RMSE, MAPE, R², uplift)
backtester  - walk-forward backtesting harness
report      - human-readable summaries, model comparison, metric export
"""

__all__ = ["backtester", "metrics", "report"]
