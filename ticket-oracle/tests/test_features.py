"""Feature engineering tests for Ticket-Oracle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TEXT_COL,
    engineer_features,
    make_feature_pipeline,
    make_synthetic_dataset,
    payload_to_dataframe,
)


def test_engineer_features_defaults():
    result = engineer_features({"hour_of_day": 12, "day_of_week": 2, "day_of_month": 15})
    assert "is_weekend" in result
    assert "is_month_end" in result
    assert result["is_weekend"] == 0


def test_engineer_features_weekend():
    result = engineer_features({"hour_of_day": 10, "day_of_week": 6, "day_of_month": 15})
    assert result["is_weekend"] == 1


def test_engineer_features_month_end():
    result = engineer_features({"hour_of_day": 9, "day_of_week": 1, "day_of_month": 31})
    assert result["is_month_end"] == 1


def test_engineer_features_not_month_end():
    result = engineer_features({"hour_of_day": 9, "day_of_week": 1, "day_of_month": 10})
    assert result["is_month_end"] == 0


def test_engineer_features_preserves_original_fields():
    raw = {"description": "Test ticket", "department": "IT", "hour_of_day": 9, "day_of_week": 0, "day_of_month": 5}
    result = engineer_features(raw)
    assert result["description"] == "Test ticket"
    assert result["department"] == "IT"


def test_payload_to_dataframe_returns_one_row():
    payload = {
        "description": "VPN not working",
        "department": "Engineering",
        "incident_type": "Network",
        "channel": "email",
        "customer_tier": "Gold",
        "product_area": "Infrastructure",
        "hour_of_day": 14,
        "day_of_week": 1,
    }
    df = payload_to_dataframe(payload)
    assert len(df) == 1


def test_payload_to_dataframe_has_all_cols():
    payload = {"description": "Issue description text here", "hour_of_day": 9}
    df = payload_to_dataframe(payload)
    for col in [TEXT_COL] + CATEGORICAL_COLS + NUMERIC_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_payload_to_dataframe_numeric_cols_are_float():
    payload = {"description": "Test", "hour_of_day": "14", "open_tickets_count": "50"}
    df = payload_to_dataframe(payload)
    for col in NUMERIC_COLS:
        assert df[col].dtype in [float, np.float64], f"Column {col} is not float"


def test_make_synthetic_dataset_shape():
    X, y_p, y_s = make_synthetic_dataset(n_samples=200)
    assert len(X) == 200
    assert len(y_p) == 200
    assert len(y_s) == 200


def test_make_synthetic_dataset_priority_labels():
    _, y_p, _ = make_synthetic_dataset(n_samples=500)
    unique = set(y_p)
    assert unique.issubset({0, 1, 2, 3})
    assert len(unique) == 4, "All four priority classes must appear in 500 samples"


def test_make_synthetic_dataset_sla_binary():
    _, _, y_s = make_synthetic_dataset(n_samples=500)
    assert set(y_s).issubset({0, 1})


def test_make_synthetic_dataset_text_col_non_empty():
    X, _, _ = make_synthetic_dataset(n_samples=100)
    assert all(len(t) > 5 for t in X[TEXT_COL])


def test_make_synthetic_dataset_reproducible():
    X1, y1, _ = make_synthetic_dataset(n_samples=100, random_state=7)
    X2, y2, _ = make_synthetic_dataset(n_samples=100, random_state=7)
    assert list(y1) == list(y2)


def test_make_feature_pipeline_returns_column_transformer():
    from sklearn.compose import ColumnTransformer
    ct = make_feature_pipeline()
    assert isinstance(ct, ColumnTransformer)


def test_feature_pipeline_transforms_data():

    X, y_p, _ = make_synthetic_dataset(n_samples=100)
    ct = make_feature_pipeline()
    Xt = ct.fit_transform(X)
    assert Xt.shape[0] == 100
    assert Xt.shape[1] > 10


@pytest.mark.parametrize("n", [50, 100, 500])
def test_synthetic_dataset_various_sizes(n):
    X, y_p, y_s = make_synthetic_dataset(n_samples=n)
    assert len(X) == n


@pytest.mark.parametrize("col", CATEGORICAL_COLS)
def test_categorical_columns_are_strings(col):
    X, _, _ = make_synthetic_dataset(n_samples=50)
    assert pd.api.types.is_string_dtype(X[col])


@pytest.mark.parametrize("col", NUMERIC_COLS)
def test_numeric_columns_are_numeric(col):
    X, _, _ = make_synthetic_dataset(n_samples=50)
    assert pd.api.types.is_numeric_dtype(X[col])
