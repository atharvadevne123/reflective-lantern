"""
Walk-forward backtesting for Price-Prophet models.

The backtester simulates production deployment by training on a growing
historical window and evaluating on the immediately following window,
mimicking the constraints of real time-series forecasting.
"""

from __future__ import annotations

import math
from typing import Any


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

    def __init__(self, model: Any, baseline: Any = None, window: int = 30) -> None:
        self.model = model
        self.window = window
        self._baseline: list[float] = list(baseline) if baseline is not None else []

    def run(
        self,
        X: list[Any],
        y_actual: list[float],
        y_model: list[float] | None = None,
        price_col: str = "base_price",
        demand_col: str = "demand",
    ) -> dict[str, Any]:
        """Execute the backtest using the simpler per-period API.

        When called with three positional sequences (*X*, *y_actual*, *y_model*)
        the method computes n_periods, baseline_revenue and model_revenue.

        When called with a list of record dicts and string column names (old API)
        the walk-forward logic applies.
        """
        if isinstance(X, list) and (not X or not isinstance(X[0], dict)):
            return self._run_simple(X, y_actual, y_model)
        return self._run_walk_forward(X, price_col, demand_col)

    def _run_simple(self, X: list[Any], y_actual: list[float], y_model: list[float] | None = None) -> dict[str, Any]:
        n = len(X)
        if n == 0:
            return {"n_periods": 0, "baseline_revenue": 0.0, "model_revenue": 0.0, "mae": 0.0}
        baseline = self._baseline
        baseline_revenue = sum(b * a for b, a in zip(baseline, y_actual, strict=False)) if baseline else 0.0
        preds = self.model.predict(X) if X else []
        model_revenue = sum(p * a for p, a in zip(preds, y_actual, strict=False)) if preds else 0.0
        mae = sum(abs(a - p) for a, p in zip(y_actual, preds, strict=False)) / n if preds else 0.0
        return {
            "n_periods": n,
            "baseline_revenue": round(baseline_revenue, 4),
            "model_revenue": round(model_revenue, 4),
            "mae": round(mae, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_xy(
        records: list[dict[str, Any]],
        price_col: str,
        demand_col: str,
    ):
        """Extract feature matrix X and target vector y from *records*.

        Features used: price (price_col), demand (demand_col), day_of_week,
        competition_price, is_weekend.  Target: revenue (price * demand).
        """
        X: list[list[float]] = []
        y: list[float] = []
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
    def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
        """Compute MAE, RMSE, R² for a single window."""
        n = len(actual)
        mae = sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / n
        mse = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / n
        rmse = math.sqrt(mse)

        mean_a = sum(actual) / n
        ss_tot = sum((a - mean_a) ** 2 for a in actual)
        ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

        return {"mae": mae, "rmse": rmse, "r_squared": r2}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _run_walk_forward(
        self,
        records: list[dict[str, Any]],
        price_col: str = "base_price",
        demand_col: str = "demand",
    ) -> dict[str, Any]:
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

        window_maes: list[float] = []
        window_rmses: list[float] = []
        window_r2s: list[float] = []
        n_windows = 0

        eval_start = self.window
        while eval_start + self.window <= n:
            train_records = records[:eval_start]
            eval_records = records[eval_start : eval_start + self.window]

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

        self._window_maes = window_maes
        self._window_rmses = window_rmses
        self._window_r2s = window_r2s

        return {
            "mae": sum(window_maes) / n_windows,
            "rmse": sum(window_rmses) / n_windows,
            "r_squared": sum(window_r2s) / n_windows,
            "n_windows": n_windows,
        }

    def per_window_metrics(self) -> list[dict[str, float]]:
        """Return per-window metrics from the last walk-forward run.

        Returns
        -------
        list[dict]
            One dict per evaluation window with keys 'mae', 'rmse',
            'r_squared'.  Empty list if no walk-forward run has been completed.
        """
        maes = getattr(self, "_window_maes", [])
        rmses = getattr(self, "_window_rmses", [])
        r2s = getattr(self, "_window_r2s", [])
        return [
            {"mae": mae, "rmse": rmse, "r_squared": r2}
            for mae, rmse, r2 in zip(maes, rmses, r2s, strict=False)
        ]

    def best_window(self) -> dict[str, float] | None:
        """Return the evaluation window with the lowest MAE from the last run.

        Returns
        -------
        dict or None
            Window metrics dict with the minimum MAE, or None if no run yet.
        """
        windows = self.per_window_metrics()
        if not windows:
            return None
        return min(windows, key=lambda w: w["mae"])
