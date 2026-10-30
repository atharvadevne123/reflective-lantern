"""Tests for advanced config utilities: reset_config, config_as_dict, _redact_url."""

from __future__ import annotations

import pytest

from app.config import _redact_url, config_as_dict, load_config, reset_config


@pytest.fixture(autouse=True)
def reset_after_test():
    yield
    reset_config()


class TestResetConfig:
    def test_reloads_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        reset_config()
        cfg = load_config()
        assert cfg.log_level == "DEBUG"

    def test_reset_clears_singleton(self) -> None:
        from app.config import get_config
        _ = get_config()
        reset_config()
        from app.config import _config
        assert _config is None


class TestConfigAsDict:
    def test_returns_dict(self) -> None:
        result = config_as_dict()
        assert isinstance(result, dict)

    def test_has_expected_keys(self) -> None:
        result = config_as_dict()
        expected = {"database_url", "model_path", "log_level", "api_rate_limit",
                    "cache_ttl_seconds", "drift_threshold", "horizon_days"}
        assert expected.issubset(set(result.keys()))

    def test_database_url_not_empty(self) -> None:
        result = config_as_dict()
        assert result["database_url"]

    def test_cors_origins_is_list(self) -> None:
        result = config_as_dict()
        assert isinstance(result["cors_origins"], list)


class TestRedactUrl:
    def test_sqlite_url_unchanged(self) -> None:
        url = "sqlite:///test.db"
        assert _redact_url(url) == url

    def test_postgres_url_with_password_redacted(self) -> None:
        url = "postgresql://user:secret@localhost:5432/mydb"
        result = _redact_url(url)
        assert "secret" not in result
        assert "***" in result

    def test_postgres_url_without_password_unchanged(self) -> None:
        url = "postgresql://user@localhost/mydb"
        result = _redact_url(url)
        assert result == url

    def test_no_scheme_unchanged(self) -> None:
        url = "localhost/db"
        assert _redact_url(url) == url

    def test_preserves_username(self) -> None:
        url = "postgresql://myuser:mypass@host/db"
        result = _redact_url(url)
        assert "myuser" in result

    @pytest.mark.parametrize("url,should_contain_stars", [
        ("postgresql://admin:password@host/db", True),
        ("sqlite:///test.db", False),
        ("mysql://root:pass@localhost/mydb", True),
    ])
    def test_redaction_parametrized(self, url: str, should_contain_stars: bool) -> None:
        result = _redact_url(url)
        assert ("***" in result) == should_contain_stars
