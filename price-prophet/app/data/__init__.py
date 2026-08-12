"""
Data sub-package for Price-Prophet.

Modules
-------
loader       – read CSV/JSON files and generate synthetic datasets
preprocessor – normalise, standardise, encode and clean raw data
features     – derive pricing-specific feature values
validator    – validate individual records and full datasets
augmentor    – add noise and bootstrap-sample for data augmentation
"""

__all__ = ["loader", "preprocessor", "features", "validator", "augmentor"]
