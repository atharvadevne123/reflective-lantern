"""Tests for feature engineering pipeline components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    FAULT_TYPES,
    DropCategoricalColumns,
    FaultTypeEncoder,
    GeoFeatureEngineer,
    InfinityNaNFixer,
    LagRollingFeatures,
    build_feature_pipeline,
    make_synthetic_dataset,
)


def _base_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "latitude": rng.uniform(-60, 60, n),
            "longitude": rng.uniform(-180, 180, n),
            "depth_km": rng.uniform(1, 100, n),
            "station_count": rng.integers(3, 30, n),
            "p_wave_amplitude": rng.uniform(0.5, 20.0, n),
            "s_wave_amplitude": rng.uniform(1.0, 40.0, n),
            "epicentral_distance_km": rng.uniform(10, 500, n),
            "fault_type": rng.choice(FAULT_TYPES, n),
        }
    )


class TestGeoFeatureEngineer:
    def test_adds_sp_amplitude_ratio(self) -> None:
        df = _base_df()
        out = GeoFeatureEngineer().fit_transform(df)
        assert "sp_amplitude_ratio" in out.columns

    def test_adds_seismic_moment_features(self) -> None:
        df = _base_df()
        out = GeoFeatureEngineer().fit_transform(df)
        assert "p_seismic_moment" in out.columns
        assert "s_seismic_moment" in out.columns

    def test_adds_station_density(self) -> None:
        df = _base_df()
        out = GeoFeatureEngineer().fit_transform(df)
        assert "station_density" in out.columns

    def test_no_inf_values(self) -> None:
        df = _base_df()
        out = GeoFeatureEngineer().fit_transform(df)
        numeric = out.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any()

    def test_row_count_preserved(self) -> None:
        df = _base_df(20)
        out = GeoFeatureEngineer().fit_transform(df)
        assert len(out) == 20


class TestFaultTypeEncoder:
    def test_drops_fault_type_column(self) -> None:
        df = _base_df()
        out = FaultTypeEncoder().fit_transform(df)
        assert "fault_type" not in out.columns

    def test_adds_seismicity_score(self) -> None:
        df = _base_df()
        out = FaultTypeEncoder().fit_transform(df)
        assert "fault_seismicity_score" in out.columns

    @pytest.mark.parametrize("ft", FAULT_TYPES)
    def test_one_hot_columns_created(self, ft: str) -> None:
        df = pd.DataFrame({"fault_type": [ft]})
        out = FaultTypeEncoder().fit_transform(df)
        assert f"fault_{ft}" in out.columns

    def test_seismicity_score_in_range(self) -> None:
        df = _base_df(50)
        out = FaultTypeEncoder().fit_transform(df)
        assert out["fault_seismicity_score"].between(0, 1).all()

    def test_reverse_has_highest_seismicity(self) -> None:
        score = FaultTypeEncoder.SEISMICITY_SCORES
        assert score["reverse"] == max(score.values())


class TestLagRollingFeatures:
    def test_adds_rolling_mean_columns(self) -> None:
        df = _base_df(20)
        out = LagRollingFeatures(windows=[3, 5]).fit_transform(df)
        assert "p_wave_amplitude_roll_mean_3" in out.columns
        assert "p_wave_amplitude_roll_mean_5" in out.columns

    def test_adds_lag_columns(self) -> None:
        df = _base_df(20)
        out = LagRollingFeatures(windows=[3]).fit_transform(df)
        assert "p_wave_amplitude_lag1" in out.columns
        assert "p_wave_amplitude_lag2" in out.columns

    def test_no_nan_values(self) -> None:
        df = _base_df(10)
        out = LagRollingFeatures(windows=[3]).fit_transform(df)
        numeric = out.select_dtypes(include="number")
        assert not numeric.isnull().any().any()

    def test_row_count_preserved(self) -> None:
        df = _base_df(15)
        out = LagRollingFeatures().fit_transform(df)
        assert len(out) == 15


class TestDropCategoricalColumns:
    def test_removes_object_columns(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        out = DropCategoricalColumns().fit(df).transform(df)
        assert "b" not in out.columns
        assert "a" in out.columns

    def test_preserves_numeric_columns(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        out = DropCategoricalColumns().fit(df).transform(df)
        assert list(out.columns) == ["x", "y"]


class TestInfinityNaNFixer:
    def test_replaces_inf_with_finite(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf], "b": [2.0, 3.0, 4.0]})
        fixer = InfinityNaNFixer().fit(df)
        out = fixer.transform(df)
        assert np.isfinite(out.values).all()

    def test_fills_nan(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        fixer = InfinityNaNFixer().fit(df)
        out = fixer.transform(df)
        assert not out.isnull().any().any()


class TestBuildFeaturePipeline:
    def test_pipeline_fits_and_transforms(self, small_dataset) -> None:
        X = small_dataset.drop(columns=["magnitude"])
        y = small_dataset["magnitude"]
        pipe = build_feature_pipeline()
        out = pipe.fit_transform(X, y)
        assert out.shape[0] == len(X)

    def test_pipeline_output_is_all_numeric(self, small_dataset) -> None:
        X = small_dataset.drop(columns=["magnitude"])
        pipe = build_feature_pipeline()
        out = pipe.fit_transform(X)
        assert np.isfinite(out).all()

    def test_pipeline_has_five_steps(self) -> None:
        pipe = build_feature_pipeline()
        assert len(pipe.steps) >= 5


class TestMakeSyntheticDataset:
    def test_returns_correct_n_samples(self) -> None:
        df = make_synthetic_dataset(n_samples=100)
        assert len(df) == 100

    def test_has_required_columns(self) -> None:
        df = make_synthetic_dataset(n_samples=50)
        for col in (
            "latitude",
            "longitude",
            "depth_km",
            "station_count",
            "p_wave_amplitude",
            "s_wave_amplitude",
            "epicentral_distance_km",
            "fault_type",
            "magnitude",
        ):
            assert col in df.columns, f"Missing column: {col}"

    def test_magnitude_in_valid_range(self) -> None:
        df = make_synthetic_dataset(n_samples=500)
        assert df["magnitude"].between(0.0, 10.0).all()

    def test_reproducible_with_same_seed(self) -> None:
        df1 = make_synthetic_dataset(n_samples=50, seed=99)
        df2 = make_synthetic_dataset(n_samples=50, seed=99)
        assert df1["magnitude"].round(4).equals(df2["magnitude"].round(4))
