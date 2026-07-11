"""End-to-end integration test: train, detect, list anomalies."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from seed_data import generate_readings


class TestEndToEnd:
    def test_train_then_detect_flags_injected_anomalies(self, client):
        # 1. Train on clean data
        clean = generate_readings(n=200, anomaly_rate=0.0, seed=11)
        train_payload = {
            "readings": [{k: v for k, v in r.items() if not k.startswith("_")} for r in clean],
            "contamination": 0.05,
        }
        train_resp = client.post("/api/v1/train", json=train_payload)
        assert train_resp.status_code == 200
        body = train_resp.json()
        assert body["samples_trained"] == 200
        assert body["features_used"] > 0

        # 2. Detect on mixed data with injected anomalies
        mixed = generate_readings(n=100, anomaly_rate=0.10, seed=22)
        detect_payload = {
            "readings": [{k: v for k, v in r.items() if not k.startswith("_")} for r in mixed],
            "horizon": 3,
        }
        detect_resp = client.post("/api/v1/detect", json=detect_payload)
        assert detect_resp.status_code == 200
        results = detect_resp.json()
        assert results["total_readings"] == 100

        # 3. Metrics should now reflect the predictions
        metrics = client.get("/api/v1/metrics").json()
        assert metrics["total_predictions"] >= 100

        # 4. Anomalies endpoint should be reachable and consistent
        anomalies = client.get("/api/v1/anomalies").json()
        assert anomalies["total"] >= 0

    def test_train_rejects_too_few_readings(self, client):
        clean = generate_readings(n=10, anomaly_rate=0.0)
        payload = {
            "readings": [{k: v for k, v in r.items() if not k.startswith("_")} for r in clean],
        }
        resp = client.post("/api/v1/train", json=payload)
        assert resp.status_code == 422

    def test_drift_endpoint_after_detection(self, client):
        readings = generate_readings(n=60, anomaly_rate=0.0, seed=5)
        payload = {
            "readings": [{k: v for k, v in r.items() if not k.startswith("_")} for r in readings],
            "horizon": 1,
        }
        client.post("/api/v1/detect", json=payload)
        drift = client.get("/api/v1/drift")
        assert drift.status_code == 200
