"""
Price optimisation for Price-Prophet.

The optimiser performs a grid search over candidate price points and
selects the one that maximises estimated revenue (price x predicted
demand) according to the supplied model.
"""

from __future__ import annotations


class PriceOptimizer:
    """Grid-search price optimiser.

    Given a trained demand/revenue model the optimiser evaluates
    *n_candidates* evenly-spaced price points between
    ``base_price * min_multiplier`` and ``base_price * max_multiplier``
    and returns the price that yields the highest predicted revenue.

    Parameters
    ----------
    model:
        A fitted pricing model that exposes a ``predict(X)`` method
        returning a list of floats.
    min_multiplier:
        Lowest price expressed as a fraction of *base_price*
        (default 0.5 -> 50 % of base).
    max_multiplier:
        Highest price expressed as a multiple of *base_price*
        (default 3.0 -> 300 % of base).
    """

    def __init__(
        self,
        model,
        min_multiplier: float = 0.5,
        max_multiplier: float = 3.0,
    ) -> None:
        self.model = model
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def revenue_at_price(self, price: float, features: list[float]) -> float:
        """Estimate revenue at a specific *price* using *features*.

        The model is expected to predict demand (or a revenue proxy) for
        the given feature vector.  Revenue is then ``price x prediction``.

        Parameters
        ----------
        price:
            Candidate price to evaluate.
        features:
            Feature vector for the product at this candidate price.

        Returns
        -------
        float
            Estimated revenue.
        """
        prediction = self.model.predict([features])[0]
        return price * prediction

    def optimize(
        self,
        base_price: float,
        features: list[float],
        n_candidates: int = 20,
    ) -> float:
        """Find the price that maximises estimated revenue.

        Evaluates *n_candidates* price points uniformly distributed
        between ``base_price * min_multiplier`` and
        ``base_price * max_multiplier``, then returns the candidate
        with the highest predicted revenue.

        Parameters
        ----------
        base_price:
            Current / reference price of the product.
        features:
            Feature vector used for model prediction.  The price itself
            is updated for each candidate, so the caller should supply
            features that do *not* already embed the price.
        n_candidates:
            Number of candidate prices to evaluate (default 20).

        Returns
        -------
        float
            Optimal price rounded to 2 decimal places.
        """
        if base_price <= 0.0:
            return round(base_price, 2)

        price_min = base_price * self.min_multiplier
        price_max = base_price * self.max_multiplier

        if n_candidates < 2:
            return round(price_min, 2)

        step = (price_max - price_min) / (n_candidates - 1)
        candidates = [price_min + i * step for i in range(n_candidates)]

        best_price = candidates[0]
        best_revenue = float("-inf")

        for candidate in candidates:
            # Replace the first feature (base_price) with the candidate price
            # so the model sees the actual candidate price in the feature vector.
            candidate_features = list(features)
            if candidate_features:
                candidate_features[0] = candidate
            rev = self.revenue_at_price(candidate, candidate_features)
            if rev > best_revenue:
                best_revenue = rev
                best_price = candidate

        return round(best_price, 2)
