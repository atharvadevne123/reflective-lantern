"""API endpoint tests for Claim-Cast."""
from __future__ import annotations

import pytest


def test_health_returns_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "db_connected" in body


def test_predict_valid_claim(client, sample_claim):
    resp = client.post("/api/v1/predict", json=sample_claim)
    assert resp.status_code == 200
    body = resp.json()
    assert body["claim_id"] == sample_claim["claim_id"]
    assert body["severity_label"] in {"low", "medium", "high", "critical"}
    assert 0 <= body["severity_score"] <= 1
    assert body["triage_priority"] in {1, 2, 3, 4}


def test_predict_high_severity_claim(client, high_severity_claim):
    resp = client.post("/api/v1/predict", json=high_severity_claim)
    assert resp.status_code == 200
    body = resp.json()
    assert body["claim_id"] == high_severity_claim["claim_id"]
    assert body["triage_priority"] in {1, 2}


def test_predict_missing_claim_id(client, sample_claim):
    bad = {k: v for k, v in sample_claim.items() if k != "claim_id"}
    resp = client.post("/api/v1/predict", json=bad)
    assert resp.status_code == 422


def test_predict_invalid_policy_type(client, sample_claim):
    bad = {**sample_claim, "policy_type": "unknown_type"}
    resp = client.post("/api/v1/predict", json=bad)
    assert resp.status_code == 422


@pytest.mark.parametrize("policy_type", ["home", "auto", "life", "commercial"])
def test_predict_all_policy_types(client, sample_claim, policy_type):
    claim = {**sample_claim, "claim_id": f"CLM-{policy_type}", "policy_type": policy_type}
    resp = client.post("/api/v1/predict", json=claim)
    assert resp.status_code == 200


def test_metrics_endpoint(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_predictions" in body
    assert "drift_alerts" in body
    assert "severity_classes" in body


def test_drift_endpoint(client):
    resp = client.get("/api/v1/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert "feature" in body
    assert "drift_detected" in body


def test_triage_summary_all_priorities(client):
    for priority in [1, 2, 3, 4]:
        resp = client.get(f"/api/v1/triage/{priority}")
        assert resp.status_code == 200
        body = resp.json()
        assert "sla_hours" in body


def test_triage_summary_invalid_priority(client):
    resp = client.get("/api/v1/triage/5")
    assert resp.status_code == 422


def test_premium_adjustment(client):
    resp = client.post("/api/v1/premium", json={
        "base_premium": 1200.0,
        "triage_priority": 2,
        "severity_label": "high",
        "claim_frequency": 0.5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["adjusted_premium"] > body["base_premium"]
    assert "recommendation" in body


def test_premium_low_risk(client):
    resp = client.post("/api/v1/premium", json={
        "base_premium": 800.0,
        "triage_priority": 4,
        "severity_label": "low",
        "claim_frequency": 0.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_multiplier"] == 1.0


def test_claim_stats_endpoint(client, sample_claim):
    client.post("/api/v1/predict", json=sample_claim)
    resp = client.get("/api/v1/claims/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "breakdown" in body
    assert "total" in body


def test_batch_predict(client, sample_claim):
    batch = [
        {**sample_claim, "claim_id": f"CLM-BATCH-{i}"}
        for i in range(3)
    ]
    resp = client.post("/api/v1/predict/batch", json=batch)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 3
    for r in results:
        assert r["severity_label"] in {"low", "medium", "high", "critical"}


def test_batch_predict_exceeds_limit(client, sample_claim):
    batch = [{**sample_claim, "claim_id": f"CLM-BIG-{i}"} for i in range(51)]
    resp = client.post("/api/v1/predict/batch", json=batch)
    assert resp.status_code == 400


def test_feature_importance_endpoint(client):
    resp = client.get("/api/v1/feature-importance")
    assert resp.status_code == 200
    body = resp.json()
    assert "feature_importance" in body
    scores = body["feature_importance"]
    assert len(scores) == 12
    assert all("feature" in s and "importance" in s for s in scores)
    # Verify sorted descending
    importances = [s["importance"] for s in scores]
    assert importances == sorted(importances, reverse=True)


def test_batch_predict_bulk(client, bulk_claims):
    resp = client.post("/api/v1/predict/batch", json=bulk_claims)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 10
    priorities = [r["triage_priority"] for r in results]
    assert all(p in {1, 2, 3, 4} for p in priorities)


def test_correlation_id_header_returned(client, sample_claim):
    resp = client.post("/api/v1/predict", json=sample_claim)
    assert "x-correlation-id" in resp.headers or "X-Correlation-ID" in resp.headers


def test_health_endpoint_response_time(client):
    import time

    start = time.time()
    resp = client.get("/api/v1/health")
    elapsed = time.time() - start
    assert resp.status_code == 200
    assert elapsed < 2.0
