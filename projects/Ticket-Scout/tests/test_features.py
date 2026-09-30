"""Feature engineering tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    PRIORITY_MAP,
    TextFeatureTransformer,
    _clean_text,
    _count_urgency,
    extract_structured_features,
)


def _make_df(n: int = 10) -> pd.DataFrame:
    subjects = [
        "VPN not connecting", "Password reset needed", "Laptop screen broken",
        "Cannot install software", "Outlook not syncing",
    ]
    bodies = [
        "Urgent: VPN drops every 10 minutes blocking all work.",
        "Account locked. Need password reset immediately.",
        "Screen flickers. Low priority.",
        "Install fails with error code. High priority.",
        "Emails not arriving since yesterday.",
    ]
    priorities = ["high", "critical", "low", "high", "medium"]
    rows = []
    for i in range(n):
        idx = i % 5
        rows.append({
            "subject": subjects[idx],
            "body": bodies[idx],
            "priority": priorities[idx],
            "created_hour": i % 24,
            "created_dow": i % 7,
            "org_size": 500,
        })
    return pd.DataFrame(rows)


def test_clean_text_removes_special_chars():
    result = _clean_text("Hello! World@#$ 123")
    assert "@" not in result
    assert "#" not in result
    assert "hello" in result


def test_clean_text_lowercases():
    assert _clean_text("URGENT ISSUE") == "urgent issue"


def test_count_urgency_detects_keywords():
    assert _count_urgency("This is URGENT and CRITICAL") >= 2


def test_count_urgency_zero_for_neutral():
    assert _count_urgency("please update my email preference") == 0


@pytest.mark.parametrize("priority,expected", [
    ("low", 0), ("medium", 1), ("high", 2), ("critical", 3)
])
def test_priority_map_values(priority, expected):
    assert PRIORITY_MAP[priority] == expected


def test_extract_structured_features_shape():
    df = _make_df(8)
    feats = extract_structured_features(df)
    assert len(feats) == 8
    assert feats.shape[1] >= 10


def test_extract_structured_features_no_nan():
    df = _make_df(10)
    feats = extract_structured_features(df)
    assert not feats.isnull().any().any()


def test_extract_structured_features_urgency_column():
    df = _make_df(5)
    feats = extract_structured_features(df)
    assert "urgency_score" in feats.columns
    assert (feats["urgency_score"] >= 0).all()


def test_text_feature_transformer_fit_transform():
    df = _make_df(20)
    transformer = TextFeatureTransformer(max_tfidf_features=50)
    transformer.fit(df)
    result = transformer.transform(df)
    assert result.shape[0] == 20
    assert result.shape[1] > 10
    assert not np.isnan(result).any()


def test_text_feature_transformer_deterministic():
    df = _make_df(10)
    t = TextFeatureTransformer(max_tfidf_features=50)
    t.fit(df)
    out1 = t.transform(df)
    out2 = t.transform(df)
    np.testing.assert_array_almost_equal(out1, out2)


def test_text_feature_transformer_new_unseen_text():
    df_train = _make_df(20)
    t = TextFeatureTransformer(max_tfidf_features=50)
    t.fit(df_train)

    df_new = pd.DataFrame([{
        "subject": "Brand new issue type never seen before xyzzy",
        "body": "Some completely novel body text here.",
        "priority": "low",
        "created_hour": 12,
        "created_dow": 3,
        "org_size": 300,
    }])
    result = t.transform(df_new)
    assert result.shape[0] == 1
    assert not np.isnan(result).any()
