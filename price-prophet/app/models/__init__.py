"""
Model sub-package for Price-Prophet.

Modules
-------
base           – abstract base class for all pricing models
linear         – sklearn LinearRegression wrapper
gradient_boost – XGBoost wrapper
ensemble       – weighted ensemble of multiple models
registry       – model registration and lookup
"""

__all__ = ["base", "linear", "gradient_boost", "ensemble", "registry"]
