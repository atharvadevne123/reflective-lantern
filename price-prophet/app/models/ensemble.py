"""
Ensemble pricing model for Price-Prophet.

Combines predictions from multiple :class:`~app.models.base.BasePricingModel`
instances via a weighted average.  Supports heterogeneous member models
(e.g. a mix of linear and gradient-boosted models).
"""

from __future__ import annotations

import logging

from app.models.base import BasePricingModel

logger = logging.getLogger(__name__)


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
        models: list[BasePricingModel],
        weights: list[float] | None = None,
    ) -> None:
        """Initialise the ensemble with *models* and optional *weights* for averaging."""
        if not models:
            raise ValueError("EnsemblePricingModel requires at least one member model.")
        self.models = list(models)
        self.weights = list(weights) if weights is not None else None
        self._fitted: bool = False
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Ensure *weights* length matches *models* length if provided."""
        if self.weights is not None and len(self.weights) != len(self.models):
            raise ValueError(
                f"weights length ({len(self.weights)}) must match models length ({len(self.models)})."
            )

    def _normalised_weights(self) -> list[float]:
        """Return a normalised weight vector (sums to 1.0)."""
        n = len(self.models)
        if self.weights is None:
            return [1.0 / n] * n
        total = sum(self.weights)
        if total == 0.0:
            return [1.0 / n] * n
        return [w / total for w in self.weights]

    def fit(self, X: list[list[float]], y: list[float]) -> None:
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
        logger.info(
            "EnsemblePricingModel fitted %d members on %d samples", len(self.models), len(y)
        )

    def predict(self, X: list[list[float]]) -> list[float]:
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
        all_members_fitted = all(
            getattr(m, "_fitted", False) or (hasattr(m, "is_fitted") and m.is_fitted())
            for m in self.models
        )
        if not self._fitted and not all_members_fitted:
            raise RuntimeError("EnsemblePricingModel is not fitted yet. Call fit() first.")

        w = self._normalised_weights()
        n_samples = len(X)
        combined = [0.0] * n_samples

        for model, weight in zip(self.models, w, strict=True):
            preds = model.predict(X)
            for i, pred in enumerate(preds):
                combined[i] += weight * pred

        return combined

    def member_count(self) -> int:
        """Return the number of member models in the ensemble."""
        return len(self.models)

    def add_model(self, model: BasePricingModel, weight: float = 1.0) -> None:
        """Append a new member model to the ensemble.

        Parameters
        ----------
        model:
            New pricing model to add.
        weight:
            Importance weight for the new model.  If *weights* were not
            originally set, equal weights are first materialised before adding.
        """
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        self.models.append(model)
        self.weights.append(weight)

    def remove_model(self, index: int) -> BasePricingModel:
        """Remove and return the member model at *index*.

        Parameters
        ----------
        index:
            Zero-based position of the model to remove.

        Returns
        -------
        BasePricingModel
            The removed model.

        Raises
        ------
        IndexError
            If *index* is out of range or would leave the ensemble empty.
        """
        if index < 0 or index >= len(self.models):
            raise IndexError(
                f"index {index} out of range for ensemble with {len(self.models)} members"
            )
        if len(self.models) == 1:
            raise IndexError("cannot remove the last model from the ensemble")
        removed = self.models.pop(index)
        if self.weights is not None:
            self.weights.pop(index)
        return removed

    def __repr__(self) -> str:
        """Return a concise string showing member model names and fit status."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        member_names = ", ".join(type(m).__name__ for m in self.models)
        return f"EnsemblePricingModel([{member_names}], {fitted_str})"
