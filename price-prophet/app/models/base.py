"""
Abstract base class for Price-Prophet pricing models.

Every concrete model must inherit from :class:`BasePricingModel` and
implement :meth:`fit` and :meth:`predict`.  The base class tracks
whether the model has been trained via the ``_fitted`` flag and exposes
an :meth:`is_fitted` helper used by serialisation and evaluation code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BasePricingModel(ABC):
    """Abstract base class for all dynamic pricing models.

    Subclasses must implement :meth:`fit` and :meth:`predict`, and must
    set ``self._fitted = True`` inside their :meth:`fit` implementation
    once training completes successfully.
    """

    _fitted: bool = False

    @abstractmethod
    def fit(self, X: List[List[float]], y: List[float]) -> None:
        """Train the model on feature matrix *X* and target vector *y*.

        Parameters
        ----------
        X:
            2-D list of shape ``(n_samples, n_features)``.
        y:
            1-D list of ``n_samples`` target values (e.g. revenue or
            demand).
        """

    @abstractmethod
    def predict(self, X: List[List[float]]) -> List[float]:
        """Generate predictions for feature matrix *X*.

        Parameters
        ----------
        X:
            2-D list of shape ``(n_samples, n_features)``.

        Returns
        -------
        list[float]
            Predicted values; one per row in *X*.
        """

    def is_fitted(self) -> bool:
        """Return whether this model has been trained.

        Returns
        -------
        bool
            ``True`` after :meth:`fit` has completed, ``False`` before.
        """
        return bool(self._fitted)

    def __repr__(self) -> str:  # noqa: D401
        """Compact string representation for logging and debugging."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_str})"
