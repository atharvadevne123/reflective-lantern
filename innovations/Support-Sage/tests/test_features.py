"""Feature engineering pipeline tests."""
import pandas as pd
import pytest

from app.features import (
    CombinedTextExtractor,
    TextMetaExtractor,
    build_feature_pipeline,
    generate_synthetic_data,
)


def test_meta_extractor_shape():
    df = pd.DataFrame([
        {"subject": "VPN issue urgent", "description": "Cannot connect to VPN error 800"},
        {"subject": "Password reset", "description": "User forgot password, needs reset"},
    ])
    extractor = TextMetaExtractor()
    result = extractor.fit_transform(df)
    assert result.shape[0] == 2
    assert result.shape[1] > 10


def test_meta_extractor_urgent_flag():
    df = pd.DataFrame([{"subject": "CRITICAL production down emergency", "description": "outage crash"}])
    extractor = TextMetaExtractor()
    result = extractor.fit_transform(df)
    assert result[0, 6] == 1.0


def test_combined_text_extractor():
    df = pd.DataFrame([{"subject": "Hello", "description": "World"}])
    extractor = CombinedTextExtractor()
    result = extractor.fit_transform(df)
    assert result == ["Hello World"]


def test_combined_text_extractor_handles_none():
    df = pd.DataFrame([{"subject": None, "description": None}])
    extractor = CombinedTextExtractor()
    result = extractor.fit_transform(df)
    assert isinstance(result[0], str)


def test_feature_pipeline_output_shape(sample_df):
    pipe = build_feature_pipeline()
    X = pipe.fit_transform(sample_df[["subject", "description"]])
    assert X.shape[0] == len(sample_df)
    assert X.shape[1] > 0


def test_synthetic_data_shape():
    df = generate_synthetic_data(n_samples=100)
    assert len(df) == 100
    assert set(["subject", "description", "priority", "category", "resolution_hours"]).issubset(df.columns)


def test_synthetic_data_priority_distribution():
    df = generate_synthetic_data(n_samples=1000)
    dist = df["priority"].value_counts(normalize=True)
    assert dist.get("critical", 0) < dist.get("medium", 0)


@pytest.mark.parametrize("n_samples", [50, 100, 500])
def test_synthetic_data_various_sizes(n_samples):
    df = generate_synthetic_data(n_samples=n_samples)
    assert len(df) == n_samples
    assert df["priority"].notna().all()
