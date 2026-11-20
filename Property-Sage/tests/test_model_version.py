"""Tests for get_model_version helper."""

import json

from app.model import get_model_version


def test_model_version_unknown(tmp_path, monkeypatch):
    import app.model as m
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "nonexistent.json")
    assert get_model_version() == "unknown"


def test_model_version_format(tmp_path, monkeypatch):
    import app.model as m
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"price_r2_mean": 0.9}))
    monkeypatch.setattr(m, "METRICS_PATH", metrics_file)
    version = get_model_version()
    assert version != "unknown"
    assert len(version) == 15  # YYYYMMDD_HHMMSS
