"""
Gradient boosting pricing model for Price-Prophet.

Wraps :class:`xgboost.XGBRegressor` in the :class:`BasePricingModel`
interface.  The xgboost import is deferred to individual methods to
allow the module to be imported in environments where xgboost is
unavailable (e.g. lightweight test containers).
"""

from __future__ import annotations

from typing import List

from app.models.base import BasePricingModel


class GradientBoostPricingModel(BasePricingModel):
    """XGBoost gradient-boosted tree regression model.

    Parameters
    ----------
    n_estimators:
        Number of boosting rounds (default 100).
    max_depth:
        Maximum depth of each decision tree (default 4).
    learning_rate:
        Step-size shrinkage used in each boosting step (default 0.1).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model = None
        self._fitted: bool = False

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        """Train the XGBoost model.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.
        y:
            Target values of length ``n_samples``.
        """
        from xgboost import XGBRegressor  # type: ignore

        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbosity=0,
        )
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: List[List[float]]) -> List[float]:
        """Generate predictions for *X*.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        list[float]

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        return self._model.predict(X).tolist()

    @property
    def feature_importances_(self) -> List[float]:
        """Feature importance scores from the fitted model.

        Returns
        -------
        list[float]
            Importance score for each feature column, or an empty list
            if the model has not been trained.
        """
        if self._model is None or not self._fitted:
            return []
        return self._model.feature_importances_.tolist()

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._fitted else "not fitted"
        return (
            f"GradientBoostPricingModel("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate}, "
            f"{fitted_str})"
        )
