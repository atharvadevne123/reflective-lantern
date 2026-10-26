"""API endpoint tests."""
from __future__ import annotations

import pytest


def test_health_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_returns_outcome(client, sample_match_payload):
    resp = client.post("/api/v1/predict", json=sample_match_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_outcome" in data
    assert data["predicted_outcome"] in ("home_win", "draw", "away_win")
    assert 0.0 <= data["home_win_prob"] <= 1.0
    assert 0.0 <= data["draw_prob"] <= 1.0
    assert 0.0 <= data["away_win_prob"] <= 1.0
    probs = data["home_win_prob"] + data["draw_prob"] + data["away_win_prob"]
    assert abs(probs - 1.0) < 0.01


@pytest.mark.parametrize("home_wins,away_wins", [(5, 0), (0, 5), (2, 2)])
def test_predict_varied_forms(client, sample_match_payload, home_wins, away_wins):
    payload = dict(sample_match_payload)
    payload["home_wins_last5"] = home_wins
    payload["home_losses_last5"] = 5 - home_wins
    payload["away_wins_last5"] = away_wins
    payload["away_losses_last5"] = 5 - away_wins
    payload["match_id"] = f"form-test-{home_wins}-{away_wins}"
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200


def test_predict_missing_required_field(client, sample_match_payload):
    payload = dict(sample_match_payload)
    del payload["home_team"]
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_player_score_endpoint(client):
    payload = {
        "player_id": "p001",
        "player_name": "Test Player",
        "team": "Team Alpha",
        "sport": "football",
        "goals_avg": 0.5,
        "assists_avg": 0.3,
        "win_rate": 0.6,
        "minutes_played_ratio": 0.9,
        "injury_days_out": 0,
    }
    resp = client.post("/api/v1/players/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["performance_score"] <= 100.0
    assert 0.0 <= data["fatigue_index"] <= 1.0


def test_metrics_endpoint(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data
    assert "model_version" in data


def test_drift_check_endpoint(client):
    import random
    random.seed(42)
    ref = [random.gauss(0, 1) for _ in range(100)]
    cur = [random.gauss(0, 1) for _ in range(100)]
    resp = client.post("/api/v1/drift", json={
        "feature_name": "home_elo",
        "reference_values": ref,
        "current_values": cur,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "ks_statistic" in data
    assert "drift_detected" in data


def test_drift_detects_shift(client):
    ref = [1.0] * 100
    cur = [10.0] * 100
    resp = client.post("/api/v1/drift", json={
        "feature_name": "home_elo",
        "reference_values": ref,
        "current_values": cur,
    })
    assert resp.status_code == 200
    assert resp.json()["drift_detected"] is True


def test_drift_summary_endpoint(client):
    resp = client.get("/api/v1/drift/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "drift_records" in data


def test_prediction_history_endpoint(client):
    resp = client.get("/api/v1/predictions/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
