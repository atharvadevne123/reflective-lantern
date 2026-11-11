"""Configuration tests for Risk-Sense."""

from __future__ import annotations

import pytest

from app.config import Settings


class TestSettings:
    def test_defaults_are_sensible(self) -> None:
        s = Settings()
        assert s.api_port == 8000
        assert s.log_level == "INFO"
        assert s.drift_ks_threshold == 0.05
        assert "sqlite" in s.database_url or "postgresql" in s.database_url

    def test_as_dict_has_expected_keys(self) -> None:
        s = Settings()
        d = s.as_dict()
        assert "api_host" in d
        assert "api_port" in d
        assert "log_level" in d
        assert "drift_ks_threshold" in d

    def test_as_dict_does_not_expose_database_url(self) -> None:
        s = Settings()
        d = s.as_dict()
        assert "database_url" not in d

    def test_env_var_overrides_port(self, monkeypatch) -> None:
        monkeypatch.setenv("API_PORT", "9000")
        s = Settings()
        assert s.api_port == 9000

    def test_env_var_overrides_log_level(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_env_var_overrides_drift_threshold(self, monkeypatch) -> None:
        monkeypatch.setenv("DRIFT_KS_THRESHOLD", "0.1")
        s = Settings()
        assert s.drift_ks_threshold == pytest.approx(0.1)

    @pytest.mark.parametrize("port", ["8000", "8080", "9000"])
    def test_valid_port_values(self, port: str, monkeypatch) -> None:
        monkeypatch.setenv("API_PORT", port)
        s = Settings()
        assert s.api_port == int(port)
