"""Extended network feature engineering tests with parametrize and boundary coverage."""
from __future__ import annotations

import numpy as np
import pytest

from app.features import (
    FEATURE_NAMES,
    FLAG_MAP,
    HIGH_RISK_SERVICES,
    PROTOCOL_MAP,
    NetworkFeatureEngineer,
)


def _minimal_flow(**overrides) -> dict:
    base = {
        "duration": 1.0,
        "src_bytes": 100,
        "dst_bytes": 200,
        "protocol_type": "tcp",
        "flag": "SF",
        "service": "http",
        "logged_in": 1,
        "count": 5,
        "serror_rate": 0.0,
        "rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "dst_host_count": 10,
    }
    base.update(overrides)
    return base


def test_transform_single_dict_returns_2d_array() -> None:
    eng = NetworkFeatureEngineer()
    result = eng.transform(_minimal_flow())
    assert result.ndim == 2
    assert result.shape[0] == 1
    assert result.shape[1] == len(FEATURE_NAMES)


def test_transform_list_of_dicts_returns_correct_shape() -> None:
    eng = NetworkFeatureEngineer()
    flows = [_minimal_flow() for _ in range(10)]
    result = eng.transform(flows)
    assert result.shape == (10, len(FEATURE_NAMES))


def test_transform_empty_list_returns_empty_matrix() -> None:
    eng = NetworkFeatureEngineer()
    result = eng.transform([])
    assert result.shape == (0, len(FEATURE_NAMES))


def test_transform_dtype_is_float32() -> None:
    eng = NetworkFeatureEngineer()
    result = eng.transform([_minimal_flow()])
    assert result.dtype == np.float32


@pytest.mark.parametrize("protocol", list(PROTOCOL_MAP.keys()))
def test_protocol_encoding_known_protocols(protocol: str) -> None:
    eng = NetworkFeatureEngineer()
    flow = _minimal_flow(protocol_type=protocol)
    result = eng.transform([flow])
    protocol_col = FEATURE_NAMES.index("protocol_encoded")
    assert result[0, protocol_col] == float(PROTOCOL_MAP[protocol])


def test_protocol_encoding_unknown_defaults_to_zero() -> None:
    eng = NetworkFeatureEngineer()
    flow = _minimal_flow(protocol_type="unknown_protocol")
    result = eng.transform([flow])
    protocol_col = FEATURE_NAMES.index("protocol_encoded")
    assert result[0, protocol_col] == 0.0


@pytest.mark.parametrize("flag", list(FLAG_MAP.keys()))
def test_flag_encoding_known_flags(flag: str) -> None:
    eng = NetworkFeatureEngineer()
    flow = _minimal_flow(flag=flag)
    result = eng.transform([flow])
    flag_col = FEATURE_NAMES.index("flag_encoded")
    assert result[0, flag_col] == float(FLAG_MAP[flag])


@pytest.mark.parametrize("service", list(HIGH_RISK_SERVICES)[:5])
def test_high_risk_service_flag_set(service: str) -> None:
    eng = NetworkFeatureEngineer()
    flow = _minimal_flow(service=service)
    result = eng.transform([flow])
    risk_col = FEATURE_NAMES.index("service_risk_score")
    assert result[0, risk_col] == 1.0


def test_low_risk_service_flag_zero() -> None:
    eng = NetworkFeatureEngineer()
    flow = _minimal_flow(service="http")
    result = eng.transform([flow])
    risk_col = FEATURE_NAMES.index("service_risk_score")
    assert result[0, risk_col] == 0.0


def test_total_bytes_sum_of_src_and_dst() -> None:
    eng = NetworkFeatureEngineer()
    flow = _minimal_flow(src_bytes=300, dst_bytes=200)
    result = eng.transform([flow])
    total_col = FEATURE_NAMES.index("total_bytes")
    assert result[0, total_col] == pytest.approx(500.0, rel=1e-5)


@pytest.mark.parametrize("n", [1, 5, 10, 50])
def test_transform_various_batch_sizes(n: int) -> None:
    eng = NetworkFeatureEngineer()
    flows = [_minimal_flow() for _ in range(n)]
    result = eng.transform(flows)
    assert result.shape[0] == n


def test_fit_returns_self() -> None:
    eng = NetworkFeatureEngineer()
    result = eng.fit([_minimal_flow()])
    assert result is eng
