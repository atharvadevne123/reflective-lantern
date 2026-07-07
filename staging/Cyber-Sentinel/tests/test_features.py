"""Tests for the feature engineering pipeline."""
from __future__ import annotations

import numpy as np
import pytest


def test_build_feature_vector_length() -> None:
    from app.features import build_feature_vector

    raw = {"protocol": "tcp", "payload_bytes": 512, "duration_ms": 100.0}
    vec = build_feature_vector(raw)
    assert len(vec) == 25


def test_build_feature_vector_all_floats() -> None:
    from app.features import build_feature_vector

    raw = {"protocol": "udp", "src_bytes": 1024}
    vec = build_feature_vector(raw)
    assert all(isinstance(v, float) for v in vec)


def test_encode_protocol_tcp() -> None:
    from app.features import encode_protocol

    assert encode_protocol("tcp") == 0


def test_encode_protocol_udp() -> None:
    from app.features import encode_protocol

    assert encode_protocol("UDP") == 1


def test_encode_protocol_icmp() -> None:
    from app.features import encode_protocol

    assert encode_protocol("icmp") == 2


def test_encode_protocol_unknown() -> None:
    from app.features import encode_protocol

    assert encode_protocol("xyz") == 3


def test_compute_byte_ratio_zero_dst() -> None:
    from app.features import compute_byte_ratio

    ratio = compute_byte_ratio(1000.0, 0.0)
    assert isinstance(ratio, float)


def test_compute_byte_ratio_equal() -> None:
    from app.features import compute_byte_ratio

    ratio = compute_byte_ratio(100.0, 100.0)
    assert abs(ratio) < 1e-6


def test_compute_entropy_uniform() -> None:
    from app.features import compute_entropy

    counts = [10, 10, 10, 10]
    entropy = compute_entropy(counts)
    assert entropy > 0


def test_compute_entropy_single() -> None:
    from app.features import compute_entropy

    counts = [100, 0, 0]
    entropy = compute_entropy(counts)
    assert entropy == 0.0


def test_compute_entropy_empty() -> None:
    from app.features import compute_entropy

    assert compute_entropy([]) == 0.0


def test_generate_synthetic_data_shape() -> None:
    from app.features import generate_synthetic_data

    X, y = generate_synthetic_data(n_samples=500)
    assert X.shape == (500, 25)
    assert y.shape == (500,)


def test_generate_synthetic_data_labels() -> None:
    from app.features import generate_synthetic_data

    _, y = generate_synthetic_data(n_samples=500)
    unique = set(y)
    assert unique.issubset({0, 1, 2, 3, 4})


def test_build_preprocessing_pipeline() -> None:
    from app.features import build_preprocessing_pipeline

    pipeline = build_preprocessing_pipeline()
    X = np.random.default_rng(0).standard_normal((100, 25))
    X_scaled = pipeline.fit_transform(X)
    assert X_scaled.shape == X.shape


@pytest.mark.parametrize("protocol", ["tcp", "udp", "icmp", "other", "UNKNOWN"])
def test_feature_vector_handles_protocols(protocol: str) -> None:
    from app.features import build_feature_vector

    vec = build_feature_vector({"protocol": protocol})
    assert len(vec) == 25


def test_feature_vector_flags_parsing() -> None:
    from app.features import build_feature_vector

    vec_syn = build_feature_vector({"flags": "S"})
    vec_fin = build_feature_vector({"flags": "F"})
    assert vec_syn != vec_fin


def test_features_from_dataframe() -> None:
    import pandas as pd

    from app.features import features_from_dataframe

    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": ["x", "y"]})
    arr = features_from_dataframe(df)
    assert arr.shape == (2, 2)
