"""
Linear regression pricing model for Price-Prophet.

Wraps scikit-learn's :class:`~sklearn.linear_model.LinearRegression` in
the :class:`BasePricingModel` interface.  The sklearn import is deferred
to individual methods so that the class can be imported and inspected in
test environments that mock the dependency.
"""

from __future__ import annotations

from app.models.base import BasePricingModel


class LinearPricingModel(BasePricingModel):
    """Ordinary least-squares linear regression pricing model.

    Parameters
    ----------
    fit_intercept:
        Whether to fit an intercept term (default ``True``).
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self._model = None
        self._fitted: bool = False

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        """Train a linear regression model.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.
        y:
            Target values of length ``n_samples``.
        """
        from sklearn.linear_model import LinearRegression  # type: ignore

        self._model = LinearRegression(fit_intercept=self.fit_intercept)
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict target values for *X*.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        list[float]
        """
        from sklearn.linear_model import (
            LinearRegression,  # noqa: F401 - ensure import path consistent
        )

        if self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        return self._model.predict(X).tolist()

    @property
    def coef_(self) -> list[float]:
        """Fitted regression coefficients.

        Returns
        -------
        list[float]
            One coefficient per feature, or an empty list if the model
            has not been trained yet.
        """
        if self._model is None or not self._fitted:
            return []
        return self._model.coef_.tolist()

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._fitted else "not fitted"
        return f"LinearPricingModel(fit_intercept={self.fit_intercept}, {fitted_str})"
