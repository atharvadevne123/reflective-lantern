"""Tests for batch prediction endpoint."""

import pytest


@pytest.mark.parametrize("batch_size", [1, 3, 5])
def test_batch_predict(client, batch_size):
    tickets = [
        {"subject": f"Issue number {i}", "description": f"Description for ticket {i} with enough detail"}
        for i in range(batch_size)
    ]
    response = client.post("/api/v1/predict/batch", json=tickets)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == batch_size
    assert len(data["results"]) == batch_size


def test_batch_predict_exceeds_limit(client):
    tickets = [
        {"subject": f"Issue {i}", "description": f"Description {i} that is long enough to pass validation"}
        for i in range(51)
    ]
    response = client.post("/api/v1/predict/batch", json=tickets)
    assert response.status_code == 200
    assert "error" in response.json()
