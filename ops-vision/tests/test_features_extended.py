"""Extended ops-vision features tests with parametrize and boundary coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from app.features import (
    FEATURE_COLS,
    clip_to_unit_interval,
    safe_dataframe_from_dict,
    validate_positive_float,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (5.0, 5.0),
        (100.0, 100.0),
    ],
)
def test_validate_positive_float_valid(value: float, expected: float) -> None:
    assert validate_positive_float(value, "field") == pytest.approx(expected)


@pytest.mark.parametrize("value", [-0.001, -1.0, -100.0])
def test_validate_positive_float_negative_raises(value: float) -> None:
    with pytest.raises(ValueError, match=">="):
        validate_positive_float(value, "field")


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_validate_positive_float_non_finite_raises(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_positive_float(bad_value, "field")


@pytest.mark.parametrize(
    "value,expected",
    [
        (-5.0, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (5.0, 1.0),
    ],
)
def test_clip_to_unit_interval_parametrized(value: float, expected: float) -> None:
    assert clip_to_unit_interval(value) == pytest.approx(expected)


def test_safe_dataframe_from_dict_valid_payload() -> None:
    payload = {col: 1.0 for col in FEATURE_COLS}
    df = safe_dataframe_from_dict(payload)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1


def test_safe_dataframe_from_dict_has_all_feature_cols() -> None:
    payload = {col: 1.0 for col in FEATURE_COLS}
    df = safe_dataframe_from_dict(payload)
    for col in FEATURE_COLS:
        assert col in df.columns


def test_safe_dataframe_from_dict_missing_fields_default_to_zero() -> None:
    df = safe_dataframe_from_dict({})
    assert df.shape[0] == 1
    for col in FEATURE_COLS:
        assert df[col].iloc[0] == 0.0


def test_safe_dataframe_from_dict_negative_value_raises() -> None:
    payload = {col: 1.0 for col in FEATURE_COLS}
    payload[FEATURE_COLS[0]] = -1.0
    with pytest.raises(ValueError):
        safe_dataframe_from_dict(payload)


def test_safe_dataframe_from_dict_non_numeric_string_raises() -> None:
    payload = {col: 1.0 for col in FEATURE_COLS}
    payload[FEATURE_COLS[0]] = "not_a_number"
    with pytest.raises(ValueError):
        safe_dataframe_from_dict(payload)


@pytest.mark.parametrize("col", FEATURE_COLS)
def test_safe_dataframe_from_dict_each_col_coerced(col: str) -> None:
    payload = {c: 1.0 for c in FEATURE_COLS}
    payload[col] = 42.0
    df = safe_dataframe_from_dict(payload)
    assert df[col].iloc[0] == pytest.approx(42.0)
