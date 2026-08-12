"""Tests for app/pricing/optimizer.py."""
from __future__ import annotations


class _ConstantModel:
    def predict(self, X):
        return [100.0] * len(X)


class _PriceAwareModel:
    """Returns demand that decreases linearly with price."""
    def predict(self, X):
        return [max(0.0, 200.0 - row[0]) for row in X]


def test_optimizer_returns_float():
    from app.pricing.optimizer import PriceOptimizer
    opt = PriceOptimizer(_ConstantModel(), min_multiplier=0.5, max_multiplier=2.0)
    price = opt.optimize(base_price=100.0, features=[100.0, 80.0, 95.0])
    assert isinstance(price, float)


def test_optimizer_within_bounds():
    from app.pricing.optimizer import PriceOptimizer
    opt = PriceOptimizer(_ConstantModel(), min_multiplier=0.5, max_multiplier=2.0)
    price = opt.optimize(base_price=100.0, features=[100.0])
    assert 50.0 <= price <= 200.0


def test_optimizer_zero_base_price():
    from app.pricing.optimizer import PriceOptimizer
    opt = PriceOptimizer(_ConstantModel())
    assert opt.optimize(0.0, [0.0]) == 0.0


def test_optimizer_revenue_at_price():
    from app.pricing.optimizer import PriceOptimizer
    opt = PriceOptimizer(_ConstantModel())
    rev = opt.revenue_at_price(50.0, [50.0])
    assert rev == 50.0 * 100.0
