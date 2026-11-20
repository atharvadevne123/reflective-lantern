"""Feature engineering tests for News-Guard."""

import pandas as pd
import pytest

from app.features import (
    TextFeatureExtractor,
    TfidfHeadlineTransformer,
    build_feature_dataframe,
    compute_feature_hash,
)


def test_text_extractor_output_shape():
    df = pd.DataFrame([{
        "headline": "SHOCKING: Government hides truth!!!",
        "body": "Some body text.",
        "source_domain": "infowars.com",
    }])
    extractor = TextFeatureExtractor()
    result = extractor.transform(df)
    assert result.shape == (1, 13)


def test_text_extractor_clickbait_score():
    df_fake = pd.DataFrame([{
        "headline": "You Won't Believe What Shocking Experts Say!!!",
        "body": "",
        "source_domain": "fake.com",
    }])
    df_real = pd.DataFrame([{
        "headline": "Federal reserve holds interest rates unchanged",
        "body": "",
        "source_domain": "reuters.com",
    }])
    ext = TextFeatureExtractor()
    fake_feats = ext.transform(df_fake)
    real_feats = ext.transform(df_real)
    assert fake_feats[0, 0] > real_feats[0, 0]


def test_text_extractor_source_flags():
    df_unreliable = pd.DataFrame([{"headline": "Test", "body": "", "source_domain": "infowars.com"}])
    df_reliable = pd.DataFrame([{"headline": "Test", "body": "", "source_domain": "reuters.com"}])
    ext = TextFeatureExtractor()
    u_feats = ext.transform(df_unreliable)
    r_feats = ext.transform(df_reliable)
    assert u_feats[0, 7] == 1.0
    assert r_feats[0, 8] == 1.0


def test_tfidf_transformer_shape():
    df = pd.DataFrame([
        {"headline": "Breaking news about politics"},
        {"headline": "Scientists discover new treatment"},
        {"headline": "SHOCKING government conspiracy revealed"},
    ])
    tfidf = TfidfHeadlineTransformer(max_features=50)
    tfidf.fit(df)
    result = tfidf.transform(df)
    assert result.shape[0] == 3
    assert 1 <= result.shape[1] <= 50


def test_tfidf_transformer_not_all_zeros():
    df = pd.DataFrame([{"headline": "Breaking news story from Reuters about economy"}])
    tfidf = TfidfHeadlineTransformer(max_features=100)
    tfidf.fit(df)
    result = tfidf.transform(df)
    assert result.sum() > 0


def test_compute_feature_hash_deterministic():
    h1 = compute_feature_hash({"headline": "test", "source": "abc"})
    h2 = compute_feature_hash({"headline": "test", "source": "abc"})
    assert h1 == h2


def test_compute_feature_hash_differs_for_different_inputs():
    h1 = compute_feature_hash({"headline": "test1"})
    h2 = compute_feature_hash({"headline": "test2"})
    assert h1 != h2


def test_build_feature_dataframe_columns():
    df = build_feature_dataframe(
        headline="Test headline",
        body="Test body",
        source_domain="example.com",
    )
    assert "headline" in df.columns
    assert "body" in df.columns
    assert "source_domain" in df.columns
    assert len(df) == 1


@pytest.mark.parametrize("caps_headline,expected_high_caps", [
    ("ALL CAPS HEADLINE HERE", True),
    ("normal lowercase headline", False),
])
def test_caps_ratio(caps_headline, expected_high_caps):
    df = pd.DataFrame([{"headline": caps_headline, "body": "", "source_domain": "test.com"}])
    ext = TextFeatureExtractor()
    feats = ext.transform(df)
    caps_ratio = feats[0, 1]
    if expected_high_caps:
        assert caps_ratio > 0.5
    else:
        assert caps_ratio < 0.3
