"""Feature engineering pipeline for real estate valuation.

The pipeline is composed of five custom sklearn-compatible transformers that
run in sequence. Each transformer adds columns to the DataFrame in place,
so downstream stages can reference upstream outputs.

Pipeline stages
---------------
1. PropertyAgeTransformer  – property_age, renovation_age
2. RatioFeatureTransformer – beds_per_bath, sqft_per_bed, price_ratio_neighborhood
3. AmenityCompositeTransformer – amenity_composite [0-1], risk_score [0-1]
4. InvestmentPotentialTransformer – investment_potential [0-10]
5. TierEncoderTransformer – size_tier [1-5], age_tier [1-5]
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

__all__ = [
    "build_feature_pipeline",
    "extract_feature_array",
    "FEATURE_COLUMNS",
    "PropertyAgeTransformer",
    "RatioFeatureTransformer",
    "AmenityCompositeTransformer",
    "InvestmentPotentialTransformer",
    "TierEncoderTransformer",
]

FEATURE_COLUMNS = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "lot_size",
    "year_built",
    "condition_score",
    "school_score",
    "transit_score",
    "walkability_score",
    "crime_rate",
    "median_neighborhood_price",
    "median_price_per_sqft",
    "avg_rental_yield",
    "listing_days",
    "renovation_age",
    "property_age",
    "beds_per_bath",
    "sqft_per_bed",
    "price_ratio_neighborhood",
    "amenity_composite",
    "investment_potential",
    "risk_score",
    "size_tier",
    "age_tier",
]


class PropertyAgeTransformer(TransformerMixin, BaseEstimator):  # noqa: N801
    """Compute property age and renovation recency features."""

    def __init__(self, reference_year: int = 2026) -> None:
        self.reference_year = reference_year

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PropertyAgeTransformer":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["property_age"] = self.reference_year - X["year_built"].clip(1800, self.reference_year)
        renovation = (
            X["renovation_year"]
            if "renovation_year" in X.columns
            else pd.Series(np.nan, index=X.index)
        )
        X["renovation_age"] = np.where(
            renovation.notna() & (renovation > 0),
            self.reference_year - renovation,
            X["property_age"],
        )
        return X


class RatioFeatureTransformer(TransformerMixin, BaseEstimator):
    """Compute ratio and interaction features."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RatioFeatureTransformer":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["beds_per_bath"] = X["bedrooms"] / (X["bathrooms"].clip(lower=0.5))
        X["sqft_per_bed"] = X["sqft"] / (X["bedrooms"].clip(lower=1))
        median_ppsf = X.get("median_price_per_sqft", pd.Series(np.ones(len(X))))
        X["price_ratio_neighborhood"] = (X["sqft"] * median_ppsf) / (
            X.get("list_price", pd.Series(np.ones(len(X)) * 1e6)).clip(lower=1)
        )
        return X


class AmenityCompositeTransformer(TransformerMixin, BaseEstimator):
    """Compute composite amenity and risk scores."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AmenityCompositeTransformer":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        school = X.get("school_score", pd.Series(np.ones(len(X)) * 5.0))
        transit = X.get("transit_score", pd.Series(np.ones(len(X)) * 5.0))
        walk = X.get("walkability_score", pd.Series(np.ones(len(X)) * 5.0))
        crime = X.get("crime_rate", pd.Series(np.ones(len(X)) * 0.5))
        X["amenity_composite"] = (school * 0.4 + transit * 0.3 + walk * 0.3) / 10.0
        X["risk_score"] = crime.clip(0, 1)
        return X


class InvestmentPotentialTransformer(TransformerMixin, BaseEstimator):
    """Compute investment potential score from rental yield and appreciation proxies."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "InvestmentPotentialTransformer":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        rental_yield = X.get("avg_rental_yield", pd.Series(np.ones(len(X)) * 0.06))
        amenity = X.get("amenity_composite", pd.Series(np.ones(len(X)) * 0.5))
        risk = X.get("risk_score", pd.Series(np.ones(len(X)) * 0.5))
        X["investment_potential"] = (rental_yield * 5.0 + amenity * 3.0 - risk * 2.0).clip(0, 10)
        return X


class TierEncoderTransformer(TransformerMixin, BaseEstimator):
    """Encode size and age into discrete tier features."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TierEncoderTransformer":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["size_tier"] = (
            pd.cut(
                X["sqft"],
                bins=[0, 800, 1500, 2500, 4000, 1e9],
                labels=[1, 2, 3, 4, 5],
            )
            .astype(float)
            .fillna(3.0)
        )
        X["age_tier"] = (
            pd.cut(
                X.get("property_age", pd.Series(np.zeros(len(X)))),
                bins=[-1, 5, 15, 30, 60, 1000],
                labels=[5, 4, 3, 2, 1],
            )
            .astype(float)
            .fillna(3.0)
        )
        return X


def build_feature_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("age", PropertyAgeTransformer()),
            ("ratios", RatioFeatureTransformer()),
            ("amenity", AmenityCompositeTransformer()),
            ("investment", InvestmentPotentialTransformer()),
            ("tiers", TierEncoderTransformer()),
        ]
    )


def extract_feature_array(df: pd.DataFrame, pipeline: Pipeline) -> np.ndarray:
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(pipeline)
        transformed = pipeline.transform(df)
    except NotFittedError:
        transformed = pipeline.fit_transform(df)
    available = [c for c in FEATURE_COLUMNS if c in transformed.columns]
    result = transformed[available].fillna(0).values
    logger.debug("Extracted %d features from %d samples", result.shape[1], result.shape[0])
    return result
