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


@pytest.mark.parametrize(
    "env_var,value,attr",
    [
        ("LOG_LEVEL", "DEBUG", "log_level"),
        ("LOG_LEVEL", "WARNING", "log_level"),
        ("COMMIT_TARGET", "90", "commit_target"),
        ("PF_FIX_TIMEOUT", "600", "pf_fix_timeout"),
    ],
)
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


def test_constants_version_is_semver() -> None:
    from config.constants import VERSION

    parts = VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_constants_smtp_ports() -> None:
    from config.constants import SMTP_PORT_SSL, SMTP_PORT_TLS

    assert SMTP_PORT_TLS == 587
    assert SMTP_PORT_SSL == 465


def test_constants_github_api_base() -> None:
    from config.constants import GITHUB_API_BASE

    assert GITHUB_API_BASE.startswith("https://")
    assert "github.com" in GITHUB_API_BASE


def test_constants_notion_model_is_string() -> None:
    from config.constants import NOTION_MODEL

    assert isinstance(NOTION_MODEL, str)
    assert len(NOTION_MODEL) > 0


def test_constants_max_repos_per_page() -> None:
    from config.constants import MAX_REPOS_PER_PAGE

    assert MAX_REPOS_PER_PAGE == 100


def test_settings_is_valid_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GH_PAT", raising=False)
    from config.settings import Settings

    s = Settings()
    assert s.is_valid() is False


def test_settings_is_valid_true_when_all_present(settings_env: None) -> None:
    from config.settings import Settings

    s = Settings()
    assert s.is_valid() is True


def test_settings_history_dir_property() -> None:
    from config.settings import Settings

    s = Settings()
    assert s.history_dir.name == "history"
    assert s.history_dir.is_dir()


@pytest.mark.parametrize(
    "key",
    [
        "anthropic_api_key",
        "gh_pat",
        "notion_api_key",
        "gmail_user",
        "gmail_app_pass",
        "log_level",
        "commit_target",
        "pf_fix_timeout",
    ],
)
def test_settings_has_attribute(settings_env: None, key: str) -> None:
    from config.settings import Settings

    s = Settings()
    assert hasattr(s, key)


def test_settings_repr_masks_sensitive_fields(settings_env: None) -> None:
    from config.settings import Settings

    s = Settings()
    r = repr(s)
    assert "sk-test" not in r
    assert "ghp_test" not in r
    assert "secret_test" not in r
    assert "***" in r


def test_settings_repr_shows_non_sensitive(settings_env: None) -> None:
    from config.settings import Settings

    s = Settings()
    r = repr(s)
    assert "test@gmail.com" in r
    assert "atharvadevne123" in r


def test_settings_dry_run_default_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRY_RUN", raising=False)
    from config.settings import Settings

    s = Settings()
    assert s.dry_run is False


def test_settings_dry_run_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRY_RUN", "1")
    from config.settings import Settings

    s = Settings()
    assert s.dry_run is True


def test_settings_report_subject_prefix_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REPORT_SUBJECT_PREFIX", raising=False)
    from config.settings import Settings

    s = Settings()
    assert s.report_subject_prefix == "[Reflective Lantern]"


def test_constants_cleanup_default_days() -> None:
    from config.constants import CLEANUP_DEFAULT_DAYS

    assert CLEANUP_DEFAULT_DAYS == 90


def test_constants_max_history_entries() -> None:
    from config.constants import MAX_HISTORY_ENTRIES

    assert MAX_HISTORY_ENTRIES > 0


def test_constants_pdf_max_size_mb() -> None:
    from config.constants import PDF_MAX_SIZE_MB

    assert PDF_MAX_SIZE_MB > 0


def test_constants_weekly_summary_days() -> None:
    from config.constants import WEEKLY_SUMMARY_DAYS

    assert WEEKLY_SUMMARY_DAYS == 7


def test_constants_min_repos_for_rotation() -> None:
    from config.constants import MIN_REPOS_FOR_ROTATION

    assert MIN_REPOS_FOR_ROTATION >= 1


def test_settings_has_dry_run_attribute() -> None:
    from config.settings import Settings

    s = Settings()
    assert hasattr(s, "dry_run")
    assert isinstance(s.dry_run, bool)


def test_settings_has_report_subject_prefix() -> None:
    from config.settings import Settings

    s = Settings()
    assert hasattr(s, "report_subject_prefix")
    assert isinstance(s.report_subject_prefix, str)


def test_email_configured_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from config.settings import Settings

    monkeypatch.delenv("GMAIL_USER", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASS", raising=False)
    s = Settings()
    assert s.email_configured() is False


def test_email_configured_true_when_set(settings_env) -> None:
    from config.settings import Settings

    s = Settings()
    assert s.email_configured() is True


def test_email_configured_false_when_only_user_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from config.settings import Settings

    monkeypatch.setenv("GMAIL_USER", "user@example.com")
    monkeypatch.delenv("GMAIL_APP_PASS", raising=False)
    s = Settings()
    assert s.email_configured() is False


def test_foundry_configured_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from config.settings import Settings

    for var in ("FOUNDRY_HOSTNAME", "FOUNDRY_TOKEN", "FOUNDRY_DATASET_RID"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.foundry_configured() is False


def test_commit_target_constant() -> None:
    from config.constants import COMMIT_TARGET

    assert COMMIT_TARGET > 0


def test_smtp_ports_distinct() -> None:
    from config.constants import SMTP_PORT_SSL, SMTP_PORT_TLS

    assert SMTP_PORT_TLS != SMTP_PORT_SSL


def test_smtp_host_is_string() -> None:
    from config.constants import SMTP_HOST

    assert isinstance(SMTP_HOST, str)
    assert len(SMTP_HOST) > 0


def test_github_owner_is_string() -> None:
    from config.constants import GITHUB_OWNER

    assert isinstance(GITHUB_OWNER, str)
    assert len(GITHUB_OWNER) > 0


def test_separator_width_positive() -> None:
    from config.constants import SEPARATOR_WIDTH

    assert SEPARATOR_WIDTH > 0


def test_weekly_summary_days_constant() -> None:
    from config.constants import WEEKLY_SUMMARY_DAYS

    assert WEEKLY_SUMMARY_DAYS > 0


@pytest.mark.parametrize("const_name", [
    "VERSION",
    "SMTP_HOST",
    "GITHUB_OWNER",
    "GITHUB_API_BASE",
    "REPORT_DATE_FORMAT",
])
def test_string_constants_are_non_empty(const_name: str) -> None:
    import config.constants as c

    val = getattr(c, const_name)
    assert isinstance(val, str)
    assert len(val) > 0
