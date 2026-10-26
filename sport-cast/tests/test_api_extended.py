"""Extended API tests with parametrized edge cases."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("elo_diff,expected_favoured", [
    (400, "home_win"),
    (-400, "away_win"),
])
def test_predict_elo_extreme_difference(client, sample_match_payload, elo_diff, expected_favoured):
    payload = dict(sample_match_payload)
    payload["match_id"] = f"elo-extreme-{elo_diff}"
    payload["home_elo"] = 1700.0
    payload["away_elo"] = 1700.0 - elo_diff
    payload["home_wins_last5"] = 4 if elo_diff > 0 else 1
    payload["away_wins_last5"] = 1 if elo_diff > 0 else 4
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_outcome"] == expected_favoured


def test_player_score_high_performer(client):
    payload = {
        "player_id": "star-001",
        "player_name": "Star Player",
        "team": "Team A",
        "sport": "football",
        "goals_avg": 1.5,
        "assists_avg": 1.0,
        "win_rate": 0.9,
        "minutes_played_ratio": 1.0,
        "injury_days_out": 0,
    }
    resp = client.post("/api/v1/players/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["performance_score"] >= 60


def test_player_score_injured_player(client):
    payload = {
        "player_id": "injured-001",
        "player_name": "Injured Player",
        "team": "Team B",
        "sport": "football",
        "goals_avg": 0.5,
        "assists_avg": 0.2,
        "win_rate": 0.4,
        "minutes_played_ratio": 0.3,
        "injury_days_out": 20,
    }
    resp = client.post("/api/v1/players/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["performance_score"] < 40


@pytest.mark.parametrize("invalid_elo", [-100.0, 0.0, 3000.0])
def test_predict_invalid_elo_rejected(client, sample_match_payload, invalid_elo):
    payload = dict(sample_match_payload)
    payload["home_elo"] = invalid_elo
    payload["match_id"] = f"invalid-elo-{invalid_elo}"
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_returns_match_id(client, sample_match_payload):
    resp = client.post("/api/v1/predict", json=sample_match_payload)
    assert resp.status_code == 200
    assert resp.json()["match_id"] == sample_match_payload["match_id"]


def test_metrics_after_multiple_predictions(client, sample_match_payload):
    for i in range(3):
        payload = dict(sample_match_payload)
        payload["match_id"] = f"metrics-test-{i}"
        client.post("/api/v1/predict", json=payload)
    resp = client.get("/api/v1/metrics")
    data = resp.json()
    assert data["total_predictions"] >= 3


def test_health_reports_model_loaded(client):
    resp = client.get("/api/v1/health")
    assert resp.json()["model_loaded"] is True
