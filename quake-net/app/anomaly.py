"""Isolation Forest anomaly detection for unusual seismic signatures."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.similarity import SIGNATURE_COLUMNS

logger = logging.getLogger(__name__)

CONTAMINATION = 0.05


class SeismicAnomalyDetector:
    """Wraps an IsolationForest with a stable, documented scoring contract."""

    def __init__(self, contamination: float = CONTAMINATION, random_state: int = 42) -> None:
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted

    def fit(self, df: pd.DataFrame) -> SeismicAnomalyDetector:
        """Fit the detector on the signature columns of ``df``."""
        self.model.fit(self._matrix(df))
        self._fitted = True
        logger.info("Anomaly detector fitted on %d events", len(df))
        return self

    def score(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Score each row, returning anomaly flags and normalised scores.

        Raises:
            ValueError: If called before :meth:`fit`.
        """
        if not self._fitted:
            raise ValueError("SeismicAnomalyDetector.score called before fit()")

        matrix = self._matrix(df)
        labels = self.model.predict(matrix)
        raw = self.model.score_samples(matrix)

        return [
            {
                "is_anomaly": bool(label == -1),
                "anomaly_score": round(float(-score), 4),
            }
            for label, score in zip(labels, raw, strict=True)
        ]

    @staticmethod
    def _matrix(df: pd.DataFrame) -> np.ndarray:
        missing = [col for col in SIGNATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing signature columns: {missing}")
        return np.log1p(np.clip(df[SIGNATURE_COLUMNS].to_numpy(dtype=float), 0.0, None))


def zscore_outliers(values: list[float], threshold: float = 3.0) -> list[bool]:
    """Flag values further than ``threshold`` standard deviations from the mean."""
    if len(values) < 3:
        return [False] * len(values)
    array = np.asarray(values, dtype=float)
    std = array.std()
    if std == 0:
        return [False] * len(values)
    return [bool(abs(z) > threshold) for z in (array - array.mean()) / std]


def iqr_outliers(values: list[float], multiplier: float = 1.5) -> list[bool]:
    """Flag values outside the Tukey fence defined by ``multiplier``."""
    if len(values) < 4:
        return [False] * len(values)
    array = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(array, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return [False] * len(values)
    low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
    return [bool(v < low or v > high) for v in array]
