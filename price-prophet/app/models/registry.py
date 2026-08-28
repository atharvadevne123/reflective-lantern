"""
Model registry for Price-Prophet.

Provides a simple named store for trained model objects so that the API
and CLI can look up models by name without importing them directly.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """A thread-safe (GIL-protected) named registry for pricing models.

    Models can be stored under arbitrary string names and retrieved by
    name later.  The registry makes no assumptions about the model type.
    """

    def __init__(self) -> None:
        """Initialise an empty model registry."""
        self._store: dict[str, Any] = {}

    def register(self, name: str, model: Any) -> None:
        """Store *model* under *name*.

        Overwrites silently if a model with the same *name* already
        exists.

        Parameters
        ----------
        name:
            Unique identifier for this model.
        model:
            Any trained or untrained model object.
        """
        self._store[name] = model
        logger.debug("registered model %r (type=%s)", name, type(model).__name__)

    def get(self, name: str) -> Any:
        """Return the model registered under *name*.

        Parameters
        ----------
        name:
            Identifier used when the model was registered.

        Returns
        -------
        Any
            The stored model object.

        Raises
        ------
        KeyError
            If no model is registered under *name*.
        """
        if name not in self._store:
            raise KeyError(f"No model registered under name {name!r}.")
        return self._store[name]

    def list_models(self) -> list[str]:
        """Return a sorted list of all registered model names.

        Returns
        -------
        list[str]
        """
        return sorted(self._store.keys())

    def remove(self, name: str) -> bool:
        """Remove the model registered under *name* if it exists.

        Parameters
        ----------
        name:
            Identifier of the model to remove.

        Returns
        -------
        bool
            ``True`` if a model was removed, ``False`` if *name* was
            not present.
        """
        if name in self._store:
            del self._store[name]
            return True
        return False

    def __contains__(self, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return name in self._store

    def __len__(self) -> int:
        """Return the number of models currently registered."""
        return len(self._store)

    def __repr__(self) -> str:
        """Return a concise string listing all registered model names."""
        names = self.list_models()
        return f"ModelRegistry({names})"


# Module-level singleton - import and use this throughout the application.
registry = ModelRegistry()
