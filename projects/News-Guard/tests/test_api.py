"""API endpoint tests for News-Guard."""

import pytest


def test_health_endpoint(api_client):
    resp = api_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_fake_article(api_client):
    payload = {
        "headline": "SHOCKING: Government Hides Truth About Vaccines!!!",
        "body": "Sources claim explosive revelations about the deep state.",
        "source_domain": "infowars.com",
        "publication_hour": 22,
        "share_count": 50000,
    }
    resp = api_client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert data["prediction"] in ("FAKE", "REAL")
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["credibility_score"] <= 1.0
    assert "article_id" in data


def test_predict_real_article(api_client):
    payload = {
        "headline": "Reuters: Federal Reserve holds rates steady",
        "body": "The Federal Reserve kept its benchmark rate unchanged in its June meeting.",
        "source_domain": "reuters.com",
        "publication_hour": 14,
        "share_count": 1200,
    }
    resp = api_client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in ("FAKE", "REAL")


def test_predict_missing_headline(api_client):
    resp = api_client.post("/api/v1/predict", json={"body": "some text"})
    assert resp.status_code == 422


def test_predict_short_headline_rejected(api_client):
    resp = api_client.post("/api/v1/predict", json={"headline": "Hi"})
    assert resp.status_code == 422


def test_metrics_endpoint(api_client):
    api_client.post("/api/v1/predict", json={
        "headline": "Test article for metrics endpoint check",
        "source_domain": "example.com",
    })
    resp = api_client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data
    assert "fake_rate" in data
    assert isinstance(data["drift_detected"], bool)


def test_drift_endpoint(api_client):
    resp = api_client.get("/api/v1/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "ks_statistic" in data or "note" in data


@pytest.mark.parametrize("hour,share_count", [
    (0, 0), (12, 500), (23, 99999),
])
def test_predict_parametrized_inputs(api_client, hour, share_count):
    resp = api_client.post("/api/v1/predict", json={
        "headline": "Breaking: Latest news story from reputable source",
        "source_domain": "bbc.com",
        "publication_hour": hour,
        "share_count": share_count,
    })
    assert resp.status_code == 200
    assert resp.json()["prediction"] in ("FAKE", "REAL")
