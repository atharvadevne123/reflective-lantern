"""Batch prediction endpoint tests."""
from __future__ import annotations


def test_batch_predict_single(client, sample_match_payload):
    resp = client.post("/api/v1/predict/batch", json={"matches": [sample_match_payload]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert len(data["results"]) == 1
    assert "predicted_outcome" in data["results"][0]


def test_batch_predict_multiple(client, sample_match_payload):
    matches = []
    for i in range(5):
        m = dict(sample_match_payload)
        m["match_id"] = f"batch-{i}"
        matches.append(m)
    resp = client.post("/api/v1/predict/batch", json={"matches": matches})
    assert resp.status_code == 200
    assert resp.json()["count"] == 5


def test_batch_predict_empty_rejected(client):
    resp = client.post("/api/v1/predict/batch", json={"matches": []})
    assert resp.status_code == 422


def test_batch_predict_probabilities_valid(client, sample_match_payload):
    resp = client.post("/api/v1/predict/batch", json={"matches": [sample_match_payload]})
    result = resp.json()["results"][0]
    total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
    assert abs(total - 1.0) < 0.01
