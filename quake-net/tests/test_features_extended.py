"""Extended feature engineering tests for quake-net with boundary and parametrize coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import FaultTypeEncoder, GeoFeatureEngineer


def _base_row(**overrides) -> dict:
    row = {
        "latitude": 35.0,
        "longitude": -120.0,
        "depth_km": 10.0,
        "station_count": 5,
        "p_wave_amplitude": 100.0,
        "s_wave_amplitude": 200.0,
        "epicentral_distance_km": 50.0,
        "fault_type": "strike_slip",
    }
    row.update(overrides)
    return row


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.mark.parametrize("n", [1, 10, 100])
def test_geo_feature_engineer_row_count(n: int) -> None:
    rng = np.random.default_rng(n)
    df = pd.DataFrame({
        "latitude": rng.uniform(-90, 90, n),
        "longitude": rng.uniform(-180, 180, n),
        "depth_km": rng.uniform(0, 300, n),
        "station_count": rng.integers(1, 50, n).astype(float),
        "p_wave_amplitude": rng.exponential(100, n),
        "s_wave_amplitude": rng.exponential(150, n),
        "epicentral_distance_km": rng.uniform(1, 500, n),
        "fault_type": "strike_slip",
    })
    eng = GeoFeatureEngineer()
    out = eng.fit_transform(df)
    assert out.shape[0] == n


def test_dist_to_equator_is_abs_latitude() -> None:
    df = _make_df([_base_row(latitude=35.0), _base_row(latitude=-35.0)])
    out = GeoFeatureEngineer().fit_transform(df)
    assert out["dist_to_equator"].iloc[0] == 35.0
    assert out["dist_to_equator"].iloc[1] == 35.0


@pytest.mark.parametrize("lat,expected", [
    (0.0, 0.0),
    (45.0, 45.0),
    (-60.0, 60.0),
    (90.0, 90.0),
])
def test_dist_to_equator_values(lat: float, expected: float) -> None:
    df = _make_df([_base_row(latitude=lat)])
    out = GeoFeatureEngineer().fit_transform(df)
    assert out["dist_to_equator"].iloc[0] == pytest.approx(expected, abs=1e-6)


def test_sp_amplitude_ratio_formula() -> None:
    df = _make_df([_base_row(p_wave_amplitude=100.0, s_wave_amplitude=200.0)])
    out = GeoFeatureEngineer().fit_transform(df)
    expected = 200.0 / (100.0 + 1e-6)
    assert out["sp_amplitude_ratio"].iloc[0] == pytest.approx(expected, rel=1e-5)


def test_sp_amplitude_ratio_zero_p_wave() -> None:
    df = _make_df([_base_row(p_wave_amplitude=0.0, s_wave_amplitude=100.0)])
    out = GeoFeatureEngineer().fit_transform(df)
    assert np.isfinite(out["sp_amplitude_ratio"].iloc[0])


def test_log_depth_is_log1p() -> None:
    df = _make_df([_base_row(depth_km=50.0)])
    out = GeoFeatureEngineer().fit_transform(df)
    assert out["log_depth"].iloc[0] == pytest.approx(np.log1p(50.0), rel=1e-6)


def test_log_depth_zero_depth() -> None:
    df = _make_df([_base_row(depth_km=0.0)])
    out = GeoFeatureEngineer().fit_transform(df)
    assert out["log_depth"].iloc[0] == pytest.approx(np.log1p(0.0), abs=1e-6)


def test_total_wave_energy_formula() -> None:
    df = _make_df([_base_row(p_wave_amplitude=3.0, s_wave_amplitude=4.0)])
    out = GeoFeatureEngineer().fit_transform(df)
    assert out["total_wave_energy"].iloc[0] == pytest.approx(5.0, rel=1e-5)


def test_station_density_formula() -> None:
    df = _make_df([_base_row(station_count=10, epicentral_distance_km=99.0)])
    out = GeoFeatureEngineer().fit_transform(df)
    expected = 10.0 / (99.0 + 1)
    assert out["station_density"].iloc[0] == pytest.approx(expected, rel=1e-5)


def test_no_nans_in_geo_features() -> None:
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "latitude": rng.uniform(-90, 90, n),
        "longitude": rng.uniform(-180, 180, n),
        "depth_km": rng.uniform(0, 300, n),
        "station_count": rng.integers(1, 50, n).astype(float),
        "p_wave_amplitude": rng.exponential(100, n),
        "s_wave_amplitude": rng.exponential(150, n),
        "epicentral_distance_km": rng.uniform(1, 500, n),
        "fault_type": "strike_slip",
    })
    out = GeoFeatureEngineer().fit_transform(df)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    assert not out[numeric_cols].isnull().any().any()


@pytest.mark.parametrize("fault_type,expected_score", [
    ("reverse", 0.9),
    ("strike_slip", 0.75),
    ("oblique", 0.6),
    ("normal", 0.5),
    ("unknown", 0.4),
])
def test_fault_type_seismicity_scores(fault_type: str, expected_score: float) -> None:
    df = _make_df([_base_row(fault_type=fault_type)])
    enc = FaultTypeEncoder()
    out = enc.fit_transform(df)
    assert out["seismicity_score"].iloc[0] == pytest.approx(expected_score, rel=1e-6)


def test_fault_type_one_hot_columns_exist() -> None:
    df = _make_df([_base_row(fault_type="reverse")])
    enc = FaultTypeEncoder()
    out = enc.fit_transform(df)
    expected_cols = [f"fault_{ft}" for ft in ["strike_slip", "reverse", "normal", "oblique", "unknown"]]
    for col in expected_cols:
        assert col in out.columns


@pytest.mark.parametrize("fault_type", ["strike_slip", "reverse", "normal", "oblique", "unknown"])
def test_fault_type_one_hot_exactly_one_active(fault_type: str) -> None:
    df = _make_df([_base_row(fault_type=fault_type)])
    enc = FaultTypeEncoder()
    out = enc.fit_transform(df)
    hot_cols = [f"fault_{ft}" for ft in ["strike_slip", "reverse", "normal", "oblique", "unknown"]]
    total = sum(out[c].iloc[0] for c in hot_cols)
    assert total == 1
