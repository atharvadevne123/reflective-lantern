"""Extended tests for quake-net app/anomaly.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.anomaly import CONTAMINATION, SeismicAnomalyDetector, zscore_outliers
from app.similarity import SIGNATURE_COLUMNS


def _df(n: int = 20) -> pd.DataFrame:
    data = {col: np.random.rand(n) * 10 for col in SIGNATURE_COLUMNS}
    return pd.DataFrame(data)


class TestSeismicAnomalyDetector:
    def test_creates_instance(self) -> None:
        assert SeismicAnomalyDetector() is not None

    def test_not_fitted_initially(self) -> None:
        assert SeismicAnomalyDetector().is_fitted is False

    def test_fitted_after_fit(self) -> None:
        det = SeismicAnomalyDetector()
        det.fit(_df(20))
        assert det.is_fitted is True

    def test_score_raises_before_fit(self) -> None:
        det = SeismicAnomalyDetector()
        with pytest.raises(ValueError):
            det.score(_df(5))

    def test_score_returns_list(self) -> None:
        det = SeismicAnomalyDetector()
        df = _df(20)
        det.fit(df)
        results = det.score(_df(5))
        assert isinstance(results, list)

    def test_score_length_matches_input(self) -> None:
        det = SeismicAnomalyDetector()
        df = _df(20)
        det.fit(df)
        results = det.score(_df(8))
        assert len(results) == 8

    def test_score_result_has_anomaly_keys(self) -> None:
        det = SeismicAnomalyDetector()
        df = _df(20)
        det.fit(df)
        for result in det.score(_df(3)):
            assert "is_anomaly" in result
            assert "anomaly_score" in result

    def test_is_anomaly_is_bool(self) -> None:
        det = SeismicAnomalyDetector()
        df = _df(30)
        det.fit(df)
        for result in det.score(_df(5)):
            assert isinstance(result["is_anomaly"], bool)

    def test_missing_column_raises(self) -> None:
        det = SeismicAnomalyDetector()
        det.fit(_df(20))
        bad_df = pd.DataFrame({"wrong_col": [1.0, 2.0]})
        with pytest.raises(ValueError):
            det.score(bad_df)


class TestZscoreOutliers:
    def test_basic_outlier_detected(self) -> None:
        values = [1.0, 1.1, 1.0, 1.2, 100.0]
        flags = zscore_outliers(values)
        assert flags[-1] is True

    def test_no_outliers(self) -> None:
        values = [1.0, 1.1, 1.0, 1.2, 1.1]
        flags = zscore_outliers(values, threshold=3.0)
        assert not any(flags)

    def test_too_few_values_no_flags(self) -> None:
        assert zscore_outliers([1.0, 2.0]) == [False, False]

    def test_constant_values_no_flags(self) -> None:
        values = [5.0] * 10
        assert not any(zscore_outliers(values))

    def test_returns_correct_length(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        flags = zscore_outliers(values)
        assert len(flags) == 5
