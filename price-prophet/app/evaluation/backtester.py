"""
Walk-forward backtesting for Price-Prophet models.

The backtester simulates production deployment by training on a growing
historical window and evaluating on the immediately following window,
mimicking the constraints of real time-series forecasting.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


class Backtester:
    """Walk-forward backtesting harness.

    For each position in the dataset the backtester trains the model on
    all data preceding that position (up to a minimum training size) and
    evaluates on the next *window* records.  This avoids look-ahead bias.

    Parameters
    ----------
    model:
        An unfitted pricing model that implements ``fit(X, y)`` and
        ``predict(X)``.  The model is re-fitted from scratch for each
        window, so it must be resettable (i.e. a fresh or re-instantiated
        model, or one whose ``fit`` replaces its internal state).
    window:
        Number of records per evaluation window (default 30).
    """

    def __init__(self, model: Any, window: int = 30) -> None:
        self.model = model
        self.window = window

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_xy(
        records: List[Dict[str, Any]],
        price_col: str,
        demand_col: str,
    ):
        """Extract feature matrix X and target vector y from *records*.

        Features used: price (price_col), demand (demand_col), day_of_week,
        competition_price, is_weekend.  Target: revenue (price × demand).
        """
        X: List[List[float]] = []
        y: List[float] = []
        for rec in records:
            price = float(rec.get(price_col, 0.0))
            demand = float(rec.get(demand_col, 0.0))
            day = float(rec.get("day_of_week", 0.0))
            comp = float(rec.get("competition_price", 0.0))
            weekend = float(bool(rec.get("is_weekend", False)))
            X.append([price, demand, day, comp, weekend])
            y.append(price * demand)  # revenue as target
        return X, y

    @staticmethod
    def _metrics(actual: List[float], predicted: List[float]) -> Dict[str, float]:
        """Compute MAE, RMSE, R² for a single window."""
        n = len(actual)
        mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n
        mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n
        rmse = math.sqrt(mse)

        mean_a = sum(actual) / n
        ss_tot = sum((a - mean_a) ** 2 for a in actual)
        ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

        return {"mae": mae, "rmse": rmse, "r_squared": r2}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        records: List[Dict[str, Any]],
        price_col: str = "base_price",
        demand_col: str = "demand",
    ) -> Dict[str, Any]:
        """Execute the walk-forward backtest.

        The minimum training set is one full *window* of records.
        Evaluation starts at position *window* and advances by *window*
        steps.

        Parameters
        ----------
        records:
            Full historical dataset in chronological order.
        price_col:
            Name of the price column in each record dict.
        demand_col:
            Name of the demand column in each record dict.

        Returns
        -------
        dict
            Aggregated metrics over all evaluation windows::

                {
                    "mae":        float,   # mean MAE across windows
                    "rmse":       float,   # mean RMSE across windows
                    "r_squared":  float,   # mean R² across windows
                    "n_windows":  int,     # number of windows evaluated
                }
        """
        n = len(records)
        if n < self.window * 2:
            # Not enough data for even one train+eval split; return NaNs
            return {"mae": float("nan"), "rmse": float("nan"), "r_squared": float("nan"), "n_windows": 0}

        window_maes: List[float] = []
        window_rmses: List[float] = []
        window_r2s: List[float] = []
        n_windows = 0

        eval_start = self.window
        while eval_start + self.window <= n:
            train_records = records[:eval_start]
            eval_records = records[eval_start: eval_start + self.window]

            X_train, y_train = self._extract_xy(train_records, price_col, demand_col)
            X_eval, y_eval = self._extract_xy(eval_records, price_col, demand_col)

            # Re-fit on training window
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_eval)

            m = self._metrics(y_eval, preds)
            window_maes.append(m["mae"])
            window_rmses.append(m["rmse"])
            window_r2s.append(m["r_squared"])
            n_windows += 1

            eval_start += self.window

        if n_windows == 0:
            return {"mae": float("nan"), "rmse": float("nan"), "r_squared": float("nan"), "n_windows": 0}

        return {
            "mae": sum(window_maes) / n_windows,
            "rmse": sum(window_rmses) / n_windows,
            "r_squared": sum(window_r2s) / n_windows,
            "n_windows": n_windows,
        }
