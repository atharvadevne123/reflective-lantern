"""
Ensemble pricing model for Price-Prophet.

Combines predictions from multiple :class:`~app.models.base.BasePricingModel`
instances via a weighted average.  Supports heterogeneous member models
(e.g. a mix of linear and gradient-boosted models).
"""

from __future__ import annotations

from typing import List, Optional

from app.models.base import BasePricingModel


class EnsemblePricingModel(BasePricingModel):
    """Weighted-average ensemble of multiple pricing models.

    Parameters
    ----------
    models:
        List of :class:`BasePricingModel` instances to combine.
        Must contain at least one model.
    weights:
        Optional importance weight for each model.  Must be the same
        length as *models*.  Weights do not need to sum to 1 — they are
        normalised internally.  ``None`` applies equal weights.

    Raises
    ------
    ValueError
        If *models* is empty.
    """

    def __init__(
        self,
        models: List[BasePricingModel],
        weights: Optional[List[float]] = None,
    ) -> None:
        if not models:
            raise ValueError(
                "EnsemblePricingModel requires at least one member model."
            )
        self.models = list(models)
        self.weights = list(weights) if weights is not None else None
        self._fitted: bool = False
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Ensure *weights* length matches *models* length if provided."""
        if self.weights is not None and len(self.weights) != len(self.models):
            raise ValueError(
                f"weights length ({len(self.weights)}) must match "
                f"models length ({len(self.models)})."
            )

    def _normalised_weights(self) -> List[float]:
        """Return a normalised weight vector (sums to 1.0)."""
        n = len(self.models)
        if self.weights is None:
            return [1.0 / n] * n
        total = sum(self.weights)
        if total == 0.0:
            return [1.0 / n] * n
        return [w / total for w in self.weights]

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        """Fit every member model on the same training data.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.
        y:
            Target values of length ``n_samples``.
        """
        for model in self.models:
            model.fit(X, y)
        self._fitted = True

    def predict(self, X: List[List[float]]) -> List[float]:
        """Return a weighted-average prediction across all member models.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        list[float]
            Weighted average of each member model's predictions.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError(
                "EnsemblePricingModel is not fitted yet. Call fit() first."
            )

        w = self._normalised_weights()
        n_samples = len(X)
        combined = [0.0] * n_samples

        for model, weight in zip(self.models, w):
            preds = model.predict(X)
            for i, pred in enumerate(preds):
                combined[i] += weight * pred

        return combined

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._fitted else "not fitted"
        member_names = ", ".join(type(m).__name__ for m in self.models)
        return f"EnsemblePricingModel([{member_names}], {fitted_str})"
