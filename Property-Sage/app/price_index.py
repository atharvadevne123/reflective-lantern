"""Neighbourhood price index utilities for Property-Sage."""

from __future__ import annotations

import statistics
from typing import Any


class NeighborhoodPriceIndex:
    """Simple in-memory neighbourhood price index."""

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}

    def add_sample(self, neighborhood: str, price: float) -> None:
        """Add a price sample for a neighbourhood.

        Args:
            neighborhood: Neighbourhood identifier.
            price: Observed property price.
        """
        self._data.setdefault(neighborhood, []).append(price)

    def get_median(self, neighborhood: str) -> float | None:
        """Return the median price for a neighbourhood.

        Args:
            neighborhood: Neighbourhood to query.

        Returns:
            Median price, or None if no samples exist.
        """
        prices = self._data.get(neighborhood)
        if not prices:
            return None
        return statistics.median(prices)

    def get_mean(self, neighborhood: str) -> float | None:
        """Return the mean price for a neighbourhood.

        Args:
            neighborhood: Neighbourhood to query.

        Returns:
            Mean price, or None if no samples exist.
        """
        prices = self._data.get(neighborhood)
        if not prices:
            return None
        return statistics.mean(prices)

    def all_neighborhoods(self) -> list[str]:
        """Return sorted list of all indexed neighbourhoods.

        Returns:
            Sorted list of neighbourhood names.
        """
        return sorted(self._data.keys())

    def index_summary(self) -> dict[str, Any]:
        """Return a summary of all indexed neighbourhoods with statistics.

        Returns:
            Dict mapping neighbourhood name to its median and sample count.
        """
        return {
            nb: {"median": self.get_median(nb), "count": len(self._data[nb])}
            for nb in self.all_neighborhoods()
        }
