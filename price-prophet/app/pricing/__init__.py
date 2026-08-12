"""
Pricing sub-package for Price-Prophet.

Modules
-------
optimizer    – price-point search that maximises predicted revenue
elasticity   – demand-elasticity estimation and application
constraints  – hard pricing guardrails (floor, ceiling, max-change)
strategy     – named pricing strategies (penetration, premium, etc.)
"""

__all__ = ["optimizer", "elasticity", "constraints", "strategy"]
