"""Tests for feature engineering pipeline."""

import numpy as np
import pytest

from app.features import (
    FEATURE_NAMES,
    NetworkFeatureEngineer,
    build_feature_pipeline,
    generate_synthetic_dataset,
)


def test_engineer_output_shape_single_record() -> None:
    eng = NetworkFeatureEngineer()
    flow = {"duration": 1.0, "src_bytes": 500, "dst_bytes": 1000}
    result = eng.transform([flow])
    assert result.shape == (1, len(FEATURE_NAMES))


def test_engineer_output_shape_batch() -> None:
    eng = NetworkFeatureEngineer()
    flows = [{"duration": i, "src_bytes": i * 100} for i in range(10)]
    result = eng.transform(flows)
    assert result.shape == (10, len(FEATURE_NAMES))


def test_engineer_bytes_ratio_feature() -> None:
    eng = NetworkFeatureEngineer()
    flow = {"src_bytes": 1000, "dst_bytes": 500}
    result = eng.transform([flow])
    bytes_ratio_idx = FEATURE_NAMES.index("bytes_ratio")
    # src=1000, dst=500 → ratio ≈ 2.0
    assert result[0, bytes_ratio_idx] == pytest.approx(1000 / (500 + 1e-9), rel=1e-3)


def test_engineer_total_bytes_feature() -> None:
    eng = NetworkFeatureEngineer()
    flow = {"src_bytes": 300, "dst_bytes": 700}
    result = eng.transform([flow])
    total_idx = FEATURE_NAMES.index("total_bytes")
    assert result[0, total_idx] == pytest.approx(1000.0, rel=1e-3)


def test_engineer_service_risk_high_risk_service() -> None:
    eng = NetworkFeatureEngineer()
    flow = {"service": "telnet"}
    result = eng.transform([flow])
    risk_idx = FEATURE_NAMES.index("service_risk_score")
    assert result[0, risk_idx] == 1.0


def test_engineer_service_risk_low_risk_service() -> None:
    eng = NetworkFeatureEngineer()
    flow = {"service": "http"}
    result = eng.transform([flow])
    risk_idx = FEATURE_NAMES.index("service_risk_score")
    assert result[0, risk_idx] == 0.0


def test_engineer_protocol_encoding() -> None:
    eng = NetworkFeatureEngineer()
    proto_idx = FEATURE_NAMES.index("protocol_encoded")
    for proto, expected in [("tcp", 0.0), ("udp", 1.0), ("icmp", 2.0)]:
        result = eng.transform([{"protocol_type": proto}])
        assert result[0, proto_idx] == expected


def test_feature_pipeline_fit_transform() -> None:
    pipe = build_feature_pipeline()
    flows = [{"duration": i, "src_bytes": i * 100, "dst_bytes": i * 50} for i in range(20)]
    result = pipe.fit_transform(flows)
    assert result.shape == (20, len(FEATURE_NAMES))
    assert not np.isnan(result).any()


def test_engineer_empty_input_returns_2d() -> None:
    """An empty batch must still yield a 2D matrix, not a 1D empty array."""
    eng = NetworkFeatureEngineer()
    result = eng.transform([])
    assert result.ndim == 2
    assert result.shape == (0, len(FEATURE_NAMES))


def test_generate_synthetic_dataset_tiny_request() -> None:
    """n_samples below one-per-class must still produce a usable dataset."""
    X, y = generate_synthetic_dataset(n_samples=1)
    assert X.ndim == 2
    assert X.shape[0] == 5  # floor of one row per class
    assert X.shape[1] == len(FEATURE_NAMES)
    assert set(y.tolist()) == {0, 1, 2, 3, 4}


def test_generate_synthetic_dataset_shape() -> None:
    X, y = generate_synthetic_dataset(n_samples=100)
    assert X.shape[0] == 100
    assert X.shape[1] == len(FEATURE_NAMES)
    assert y.shape[0] == 100


def test_generate_synthetic_dataset_classes() -> None:
    X, y = generate_synthetic_dataset(n_samples=500)
    unique_classes = set(y.tolist())
    assert unique_classes == {0, 1, 2, 3, 4}


@pytest.mark.parametrize("flag,expected_idx", [
    ("SF", 0), ("S0", 1), ("REJ", 2), ("RSTO", 3),
])
def test_flag_encoding(flag: str, expected_idx: int) -> None:
    eng = NetworkFeatureEngineer()
    flag_col = FEATURE_NAMES.index("flag_encoded")
    result = eng.transform([{"flag": flag}])
    assert result[0, flag_col] == float(expected_idx)
