"""Model training and prediction tests for News-Guard."""

import pytest

from app.model import NewsGuardModel, generate_synthetic_data


def test_synthetic_data_generation():
    df, y = generate_synthetic_data(n_samples=100)
    assert len(df) == 100
    assert len(y) == 100
    assert set(y).issubset({0, 1})
    assert "headline" in df.columns
    assert "source_domain" in df.columns


def test_model_trains_successfully(trained_model):
    assert trained_model._inner_pipeline is not None


def test_model_auc_above_threshold(trained_model):
    import json
    from pathlib import Path
    metrics = json.loads(Path("/tmp/test_metrics.json").read_text())
    assert metrics["auc_mean"] >= 0.55


def test_predict_returns_valid_schema(trained_model, sample_article_df):
    result = trained_model.predict(sample_article_df)
    assert "prediction" in result
    assert result["prediction"] in ("FAKE", "REAL")
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["credibility_score"] <= 1.0
    assert 0.0 <= result["fake_probability"] <= 1.0


def test_predict_probabilities_sum_to_one(trained_model, sample_article_df):
    result = trained_model.predict(sample_article_df)
    total = result["fake_probability"] + result["real_probability"]
    assert abs(total - 1.0) < 1e-4


def test_model_load_creates_instance():
    m = NewsGuardModel.load()
    assert isinstance(m, NewsGuardModel)


@pytest.mark.parametrize("source,expected_label", [
    ("infowars.com", "FAKE"),
    ("reuters.com", "REAL"),
])
def test_predict_known_sources(trained_model, source, expected_label):
    import pandas as pd
    df = pd.DataFrame([{
        "headline": "SHOCKING BOMBSHELL!!! Secret they don't want you to know",
        "body": "Explosive revelations about the hidden truth.",
        "source_domain": source,
        "publication_hour": 22,
        "share_count": 50000,
    }])
    result = trained_model.predict(df)
    assert result["prediction"] in ("FAKE", "REAL")


def test_untrained_model_raises():
    m = NewsGuardModel()
    import pandas as pd
    df = pd.DataFrame([{
        "headline": "Test headline",
        "body": "",
        "source_domain": "test.com",
        "publication_hour": 12,
        "share_count": 0,
    }])
    with pytest.raises(RuntimeError):
        m.predict(df)
