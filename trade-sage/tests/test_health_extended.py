"""Extended tests for health check utilities."""

from __future__ import annotations

import pytest

from app.health import (
    HealthStatus,
    check_database,
    check_env_vars,
    liveness_probe,
)


class TestCheckDatabase:
    def test_empty_url_returns_misconfigured(self) -> None:
        assert check_database("") == "misconfigured"

    def test_valid_url_returns_ok(self) -> None:
        assert check_database("sqlite:///./test.db") == "ok"

    @pytest.mark.parametrize("url", [
        "postgresql://user:pass@host/db",
        "sqlite:///test.db",
        "mysql://localhost/mydb",
    ])
    def test_various_urls_return_ok(self, url: str) -> None:
        assert check_database(url) == "ok"


class TestCheckEnvVars:
    def test_set_vars_return_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "value")
        result = check_env_vars(["MY_VAR"])
        assert result["MY_VAR"] == "set"

    def test_unset_vars_return_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        result = check_env_vars(["NONEXISTENT_VAR_XYZ"])
        assert result["NONEXISTENT_VAR_XYZ"] == "missing"

    def test_empty_list_returns_empty_dict(self) -> None:
        assert check_env_vars([]) == {}

    def test_mixed_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SET_VAR", "1")
        monkeypatch.delenv("UNSET_VAR_XYZ", raising=False)
        result = check_env_vars(["SET_VAR", "UNSET_VAR_XYZ"])
        assert result["SET_VAR"] == "set"
        assert result["UNSET_VAR_XYZ"] == "missing"


class TestLivenessProbe:
    def test_returns_alive_status(self) -> None:
        result = liveness_probe()
        assert result["status"] == "alive"

    def test_has_uptime_seconds_key(self) -> None:
        result = liveness_probe()
        assert "uptime_seconds" in result

    def test_uptime_is_non_negative(self) -> None:
        result = liveness_probe()
        assert float(result["uptime_seconds"]) >= 0.0


class TestHealthStatus:
    def test_healthy_returns_true(self) -> None:
        hs = HealthStatus(status="healthy")
        assert hs.is_healthy() is True

    def test_degraded_returns_false(self) -> None:
        hs = HealthStatus(status="degraded")
        assert hs.is_healthy() is False

    def test_to_dict_has_status(self) -> None:
        hs = HealthStatus(status="healthy")
        d = hs.to_dict()
        assert "status" in d
        assert "checks" in d
        assert "uptime_seconds" in d
