"""Model training and prediction tests."""
import pytest


def test_train_model_produces_metrics(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.model import train_model
    _, metrics = train_model(sample_df)
    assert "auc_mean" in metrics
    assert 0.0 <= metrics["auc_mean"] <= 1.0
    assert metrics["n_samples"] == len(sample_df)
    assert metrics["n_features"] > 0


def test_model_saved_to_disk(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.model import train_model
    train_model(sample_df)
    assert (tmp_path / "models" / "priority_model.joblib").exists()
    assert (tmp_path / "models" / "metrics.json").exists()


def test_predict_returns_valid_priority(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.model import load_model, predict, train_model
    train_model(sample_df)
    model, encoder, regressor = load_model()
    result = predict("VPN not working", "Cannot connect from home, error 800", model, encoder, regressor)
    assert result["priority"] in ["critical", "high", "medium", "low"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["resolution_hours_estimate"] > 0


def test_predict_probabilities_sum_to_one(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.model import load_model, predict, train_model
    train_model(sample_df)
    model, encoder, regressor = load_model()
    result = predict("Test subject line", "Test description that is long enough to be valid input", model, encoder, regressor)
    prob_sum = sum(result["priority_probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4


def test_load_model_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.model import load_model
    with pytest.raises(FileNotFoundError):
        load_model()


@pytest.mark.parametrize("subject,desc", [
    ("URGENT: Production down", "Critical outage emergency affecting all prod users crash"),
    ("Password reset", "User needs a password reset for their account standard request"),
    ("Software install request", "Please install Python on my laptop for data analysis work"),
])
def test_predict_various_inputs(sample_df, tmp_path, monkeypatch, subject, desc):
    monkeypatch.chdir(tmp_path)
    from app.model import load_model, predict, train_model
    train_model(sample_df)
    model, encoder, regressor = load_model()
    result = predict(subject, desc, model, encoder, regressor)
    assert result["priority"] in ["critical", "high", "medium", "low"]


def test_predict_empty_description_raises(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.model import load_model, predict, train_model
    train_model(sample_df)
    model, encoder, regressor = load_model()
    result = predict("", "", model, encoder, regressor)
    assert result["priority"] in ["critical", "high", "medium", "low"]
