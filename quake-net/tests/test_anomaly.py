"""Tests for Isolation Forest anomaly detection and statistical outlier rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.anomaly import SeismicAnomalyDetector, iqr_outliers, zscore_outliers
from app.similarity import SIGNATURE_COLUMNS


def _frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "depth_km": rng.uniform(5, 40, n),
            "p_wave_amplitude": rng.uniform(1, 10, n),
            "s_wave_amplitude": rng.uniform(2, 20, n),
            "epicentral_distance_km": rng.uniform(20, 300, n),
            "station_count": rng.integers(5, 30, n),
        }
    )


class TestSeismicAnomalyDetector:
    def test_starts_unfitted(self) -> None:
        assert SeismicAnomalyDetector().is_fitted is False

    def test_fit_marks_as_fitted(self) -> None:
        assert SeismicAnomalyDetector().fit(_frame()).is_fitted is True

    def test_score_before_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="before fit"):
            SeismicAnomalyDetector().score(_frame())

    def test_score_returns_one_row_per_event(self) -> None:
        frame = _frame(40)
        scores = SeismicAnomalyDetector().fit(frame).score(frame)
        assert len(scores) == 40

    def test_score_entries_have_expected_keys(self) -> None:
        frame = _frame(30)
        entry = SeismicAnomalyDetector().fit(frame).score(frame)[0]
        assert set(entry) == {"is_anomaly", "anomaly_score"}

    def test_is_anomaly_is_bool(self) -> None:
        frame = _frame(30)
        scores = SeismicAnomalyDetector().fit(frame).score(frame)
        assert all(isinstance(s["is_anomaly"], bool) for s in scores)

    def test_flags_extreme_outlier(self) -> None:
        frame = _frame(80)
        frame.loc[len(frame)] = {
            "depth_km": 690.0,
            "p_wave_amplitude": 5000.0,
            "s_wave_amplitude": 9000.0,
            "epicentral_distance_km": 19000.0,
            "station_count": 400,
        }
        scores = SeismicAnomalyDetector(contamination=0.05).fit(frame).score(frame)
        assert scores[-1]["is_anomaly"] is True

    def test_missing_signature_column_raises(self) -> None:
        frame = _frame(20).drop(columns=["depth_km"])
        with pytest.raises(ValueError, match="Missing signature columns"):
            SeismicAnomalyDetector().fit(frame)

    def test_extra_columns_are_ignored(self) -> None:
        frame = _frame(30)
        frame["irrelevant"] = "text"
        scores = SeismicAnomalyDetector().fit(frame).score(frame)
        assert len(scores) == 30

    @pytest.mark.parametrize("contamination", [0.01, 0.05, 0.1, 0.2])
    def test_contamination_settings_accepted(self, contamination: float) -> None:
        frame = _frame(60)
        detector = SeismicAnomalyDetector(contamination=contamination).fit(frame)
        assert len(detector.score(frame)) == 60

    def test_signature_columns_used(self) -> None:
        assert "depth_km" in SIGNATURE_COLUMNS


class TestZscoreOutliers:
    def test_flags_clear_outlier(self) -> None:
        flags = zscore_outliers([1.0] * 30 + [500.0], threshold=3.0)
        assert flags[-1] is True

    def test_no_flags_for_uniform_values(self) -> None:
        assert not any(zscore_outliers([5.0] * 20))

    def test_short_input_returns_all_false(self) -> None:
        assert zscore_outliers([1.0, 2.0]) == [False, False]

    def test_length_preserved(self) -> None:
        assert len(zscore_outliers([float(i) for i in range(25)])) == 25

    def test_zero_variance_returns_all_false(self) -> None:
        assert not any(zscore_outliers([3.0, 3.0, 3.0, 3.0]))


class TestIqrOutliers:
    def test_flags_clear_outlier(self) -> None:
        flags = iqr_outliers([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
        assert flags[-1] is True

    def test_no_flags_for_tight_cluster(self) -> None:
        assert not any(iqr_outliers([10.0, 10.1, 10.2, 10.3, 10.4]))

    def test_short_input_returns_all_false(self) -> None:
        assert iqr_outliers([1.0, 2.0, 3.0]) == [False, False, False]

    def test_length_preserved(self) -> None:
        assert len(iqr_outliers([float(i) for i in range(20)])) == 20

    def test_zero_iqr_returns_all_false(self) -> None:
        assert not any(iqr_outliers([7.0] * 10))

    @pytest.mark.parametrize("multiplier", [1.5, 3.0])
    def test_larger_multiplier_flags_no_more(self, multiplier: float) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 50.0]
        tight = sum(iqr_outliers(values, multiplier=1.5))
        loose = sum(iqr_outliers(values, multiplier=multiplier))
        assert loose <= tight


class TestZscoreOutliersParametrized:
    @pytest.mark.parametrize("threshold", [2.0, 3.0, 4.0])
    def test_extreme_value_flagged_at_strict_threshold(self, threshold: float) -> None:
        """A value 10 std-devs away is an outlier at any reasonable threshold."""
        data = [1.0] * 20 + [100.0]
        flags = zscore_outliers(data, threshold=threshold)
        assert flags[-1] is True

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_identical_values_all_false(self, n: int) -> None:
        """Identical values produce zero z-scores so no outliers are flagged."""
        flags = zscore_outliers([3.0] * n)
        assert not any(flags)

    def test_output_length_matches_input(self) -> None:
        data = list(range(15))
        assert len(zscore_outliers(data)) == len(data)
