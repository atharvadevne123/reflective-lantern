"""Tests for config.settings and config.constants."""

from __future__ import annotations

import pytest


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GH_PAT", raising=False)
    from config.settings import Settings
    s = Settings()
    assert s.anthropic_api_key == ""
    assert s.gh_pat == ""
    assert s.log_level == "INFO"
    assert s.commit_target == 60
    assert s.pf_fix_timeout == 300


def test_settings_from_env(settings_env: None) -> None:
    from config.settings import Settings
    s = Settings()
    assert s.anthropic_api_key == "sk-test"
    assert s.gh_pat == "ghp_test"
    assert s.notion_api_key == "secret_test"
    assert s.gmail_user == "test@gmail.com"


def test_settings_validate_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GH_PAT", raising=False)
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.delenv("GMAIL_USER", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASS", raising=False)
    from config.settings import Settings
    s = Settings()
    missing = s.validate()
    assert "ANTHROPIC_API_KEY" in missing
    assert "GH_PAT" in missing
    assert "NOTION_API_KEY" in missing


def test_settings_validate_all_present(settings_env: None) -> None:
    from config.settings import Settings
    s = Settings()
    assert s.validate() == []


def test_settings_json_logs_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JSON_LOGS", "1")
    from config.settings import Settings
    s = Settings()
    assert s.json_logs is True


def test_settings_json_logs_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JSON_LOGS", raising=False)
    from config.settings import Settings
    s = Settings()
    assert s.json_logs is False


def test_settings_is_frozen(settings_env: None) -> None:
    from config.settings import Settings
    s = Settings()
    with pytest.raises((AttributeError, TypeError)):
        s.log_level = "DEBUG"  # type: ignore[misc]


def test_constants_dirs_exist() -> None:
    from config.constants import HISTORY_DIR, ROOT_DIR, SCRIPTS_DIR
    assert ROOT_DIR.is_dir()
    assert HISTORY_DIR.is_dir()
    assert SCRIPTS_DIR.is_dir()


def test_constants_commit_target() -> None:
    from config.constants import COMMIT_TARGET
    assert COMMIT_TARGET == 60


def test_get_settings_singleton() -> None:
    import config.settings as cs
    cs._settings = None  # reset singleton
    s1 = cs.get_settings()
    s2 = cs.get_settings()
    assert s1 is s2


@pytest.mark.parametrize("env_var,value,attr", [
    ("LOG_LEVEL", "DEBUG", "log_level"),
    ("LOG_LEVEL", "WARNING", "log_level"),
    ("COMMIT_TARGET", "90", "commit_target"),
    ("PF_FIX_TIMEOUT", "600", "pf_fix_timeout"),
])
def test_settings_env_overrides(
    monkeypatch: pytest.MonkeyPatch, env_var: str, value: str, attr: str
) -> None:
    monkeypatch.setenv(env_var, value)
    import importlib
    import config.settings as cs
    importlib.reload(cs)
    cs._settings = None
    from config.settings import Settings
    s = Settings()
    expected = int(value) if attr in ("commit_target", "pf_fix_timeout") else value
    assert getattr(s, attr) == expected


def test_settings_report_recipient_defaults_to_gmail_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMAIL_USER", "fallback@example.com")
    monkeypatch.delenv("REPORT_RECIPIENT", raising=False)
    from config.settings import Settings
    s = Settings()
    assert s.report_recipient == "fallback@example.com"


def test_settings_report_recipient_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMAIL_USER", "sender@example.com")
    monkeypatch.setenv("REPORT_RECIPIENT", "boss@example.com")
    from config.settings import Settings
    s = Settings()
    assert s.report_recipient == "boss@example.com"


def test_constants_innovation_day_ranges() -> None:
    from config.constants import INNOVATION_DAY_RANGES, INNOVATION_WEEKDAY
    assert INNOVATION_WEEKDAY == 3
    assert (8, 14) in INNOVATION_DAY_RANGES
    assert (22, 28) in INNOVATION_DAY_RANGES


def test_non_record_files_constant() -> None:
    from config.constants import NON_RECORD_FILES
    assert "schema.json" in NON_RECORD_FILES
    assert "commit_schedule.json" in NON_RECORD_FILES
