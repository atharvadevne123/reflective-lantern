"""Feature engineering pipeline for property valuation models.

This module builds all derived features used by the ensemble model:
age, age-squared, sqft-per-bedroom, bath-bed ratio, lot density,
neighbourhood price index, and label-encoded categoricals.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

PROPERTY_TYPES: list[str] = ["apartment", "house", "condo", "townhouse", "studio"]
NEIGHBORHOODS: list[str] = [
    "downtown", "suburb", "midtown", "uptown", "waterfront",
    "historic", "industrial", "university", "airport", "rural",
]

REFERENCE_YEAR: int = 2024


class PropertyFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transform raw property attributes into model-ready features.

    Args:
        None

    Attributes:
        neighborhood_price_map: Per-neighbourhood mean sqft from training data.
        type_encoder: LabelEncoder fitted on ``property_type`` column.
        neighborhood_encoder: LabelEncoder fitted on ``neighborhood`` column.
    """

    def __init__(self) -> None:
        self.neighborhood_price_map: dict[str, float] = {}
        self.type_encoder: LabelEncoder = LabelEncoder()
        self.neighborhood_encoder: LabelEncoder = LabelEncoder()
        self._fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PropertyFeatureEngineer":
        """Fit encoders and neighbourhood index on training data.

        Args:
            X: Raw property DataFrame.
            y: Ignored; present for sklearn API compatibility.

        Returns:
            self
        """
        self.neighborhood_price_map = (
            X.groupby("neighborhood")["sqft"].mean().to_dict()
            if "sqft" in X.columns
            else {}
        )
        self.type_encoder.fit(X["property_type"].fillna("apartment"))
        self.neighborhood_encoder.fit(X["neighborhood"].fillna("suburb"))
        self._fitted = True
        logger.debug("PropertyFeatureEngineer fitted on %d rows", len(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature derivations to a property DataFrame.

        Args:
            X: Raw property DataFrame (must contain same columns as fit data).

        Returns:
            DataFrame with exactly the columns listed by :func:`get_feature_names`.
        """
        df = X.copy()

        df["property_age"] = REFERENCE_YEAR - df["year_built"].clip(upper=REFERENCE_YEAR)
        df["property_age_sq"] = df["property_age"] ** 2
        df["sqft_per_bedroom"] = df["sqft"] / (df["bedrooms"].clip(lower=1))
        df["bath_bed_ratio"] = df["bathrooms"] / (df["bedrooms"].clip(lower=1))

        lot = df.get("lot_size", pd.Series(np.ones(len(df)) * 5000, index=df.index))
        df["lot_density"] = df["sqft"] / lot.clip(lower=1)

        hood_mean = df["neighborhood"].map(self.neighborhood_price_map).fillna(df["sqft"].mean())
        df["sqft_vs_neighborhood"] = df["sqft"] / hood_mean.clip(lower=1)

        df["property_type_enc"] = self._safe_encode(self.type_encoder, df["property_type"])
        df["neighborhood_enc"] = self._safe_encode(self.neighborhood_encoder, df["neighborhood"])

        feature_cols = get_feature_names()
        return df[feature_cols].astype(float)

    def _safe_encode(self, encoder: LabelEncoder, series: pd.Series) -> pd.Series:
        """Encode a categorical series, mapping unseen labels to the first class.

        Args:
            encoder: A fitted LabelEncoder.
            series: Categorical series to encode.

        Returns:
            Integer-encoded pandas Series.
        """
        known = set(encoder.classes_)
        mapped = series.fillna(encoder.classes_[0]).map(
            lambda v: v if v in known else encoder.classes_[0]
        )
        return pd.Series(encoder.transform(mapped), index=series.index)


def build_feature_pipeline() -> Pipeline:
    """Return a full sklearn pipeline with feature engineering and scaling.

    Returns:
        sklearn Pipeline: ``features`` → ``scaler`` steps.
    """
    return Pipeline([
        ("features", PropertyFeatureEngineer()),
        ("scaler", StandardScaler()),
    ])


def generate_synthetic_data(
    n: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate synthetic property data for model training and testing.

    Args:
        n: Number of records to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (features DataFrame, price Series, rental_yield Series).
    """
    rng = np.random.default_rng(seed)

    neighborhoods = rng.choice(NEIGHBORHOODS, n)
    property_types = rng.choice(PROPERTY_TYPES, n)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n)
    sqft = rng.integers(500, 4000, n).astype(float)
    lot_size = sqft * rng.uniform(1.5, 5.0, n)
    year_built = rng.integers(1950, 2023, n)

    neighborhood_multipliers: dict[str, float] = {
        "waterfront": 1.8, "downtown": 1.5, "midtown": 1.3, "uptown": 1.2,
        "historic": 1.1, "suburb": 1.0, "university": 0.95, "airport": 0.85,
        "industrial": 0.75, "rural": 0.65,
    }
    type_multipliers: dict[str, float] = {
        "house": 1.3, "townhouse": 1.1, "condo": 1.0, "apartment": 0.9, "studio": 0.7,
    }

    base_price = 200_000
    prices = np.array([
        base_price
        * (sqft[i] / 1000)
        * neighborhood_multipliers[neighborhoods[i]]
        * type_multipliers[property_types[i]]
        * (0.98 ** max(0, 2024 - year_built[i]))
        * rng.uniform(0.85, 1.15)
        for i in range(n)
    ])

    rental_yields = np.array([
        max(0.02, min(0.12,
            0.05 * neighborhood_multipliers[neighborhoods[i]]
            * rng.uniform(0.9, 1.1)
        ))
        for i in range(n)
    ])

    df = pd.DataFrame({
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "lot_size": lot_size,
        "year_built": year_built,
        "neighborhood": neighborhoods,
        "property_type": property_types,
    })

    logger.info("Generated %d synthetic property records (seed=%d)", n, seed)
    return df, pd.Series(prices, name="price"), pd.Series(rental_yields, name="rental_yield")


def get_feature_names() -> list[str]:
    """Return the ordered list of feature column names produced by the pipeline.

    Returns:
        List of 11 feature column name strings.
    """
    return [
        "bedrooms", "bathrooms", "sqft", "property_age", "property_age_sq",
        "sqft_per_bedroom", "bath_bed_ratio", "lot_density",
        "sqft_vs_neighborhood", "property_type_enc", "neighborhood_enc",
    ]


def property_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """Convert a single property dict into a one-row DataFrame.

    Args:
        data: Dict with keys matching the PropertyRequest schema.

    Returns:
        Single-row DataFrame suitable for model inference.
    """
    return pd.DataFrame([{
        "bedrooms": data["bedrooms"],
        "bathrooms": data["bathrooms"],
        "sqft": data["sqft"],
        "lot_size": data.get("lot_size", 5000.0),
        "year_built": data["year_built"],
        "neighborhood": data["neighborhood"],
        "property_type": data["property_type"],
    }])
