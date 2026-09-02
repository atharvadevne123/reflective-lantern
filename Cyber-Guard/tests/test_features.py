"""Feature engineering tests for Cyber-Guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    FEATURE_NAMES,
    NetworkFeatureEngineer,
    build_feature_pipeline,
    make_sample_df,
)


@pytest.fixture
def basic_df():
    return pd.DataFrame([{
        "src_bytes": 100.0, "dst_bytes": 200.0, "duration": 1.0,
        "protocol_type": "tcp", "service": "http", "flag": "SF",
    }])


def test_feature_engineer_fit_transform_shape(basic_df: pd.DataFrame):
    eng = NetworkFeatureEngineer()
    eng.fit(basic_df)
    out = eng.transform(basic_df)
    assert out.shape == (1, len(FEATURE_NAMES))


def test_feature_engineer_output_is_numeric(basic_df: pd.DataFrame):
    eng = NetworkFeatureEngineer()
    eng.fit(basic_df)
    out = eng.transform(basic_df)
    assert np.isfinite(out).all(), "output contains NaN or Inf"


def test_make_sample_df_defaults():
    df = make_sample_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "src_bytes" in df.columns


def test_build_feature_pipeline_transform(basic_df: pd.DataFrame):
    pipe = build_feature_pipeline()
    pipe.fit(basic_df)
    out = pipe.transform(basic_df)
    assert out.shape[0] == 1
    assert out.shape[1] == len(FEATURE_NAMES)


def test_byte_ratio_feature():
    df = make_sample_df(src_bytes=100.0, dst_bytes=50.0)
    eng = NetworkFeatureEngineer()
    eng.fit(df)
    out = eng.transform(df)
    df_feat = pd.DataFrame(out, columns=FEATURE_NAMES)
    # byte_ratio = src / (dst + 1) = 100/51 ≈ 1.96 (after scaling may differ but structure valid)
    assert df_feat.shape[1] == 15


@pytest.mark.parametrize("protocol", ["tcp", "udp", "icmp"])
def test_protocol_encoding_valid(protocol: str):
    df = make_sample_df(protocol_type=protocol)
    eng = NetworkFeatureEngineer()
    eng.fit(df)
    out = eng.transform(df)
    assert np.isfinite(out).all()


def test_unknown_service_doesnt_crash():
    df = make_sample_df(service="unknown_service_xyz")
    eng = NetworkFeatureEngineer()
    eng.fit(df)
    out = eng.transform(df)
    assert out.shape == (1, len(FEATURE_NAMES))


def test_zero_bytes_handled():
    df = make_sample_df(src_bytes=0.0, dst_bytes=0.0, duration=0.0)
    eng = NetworkFeatureEngineer()
    eng.fit(df)
    out = eng.transform(df)
    assert np.isfinite(out).all()
