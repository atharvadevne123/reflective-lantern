"""Runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present (does not override existing env vars)
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable settings object built from environment variables.

    All values are resolved once at instantiation. Use Settings() to
    create a fresh instance (e.g. inside tests with monkeypatched env).
    """

    # Anthropic
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )

    # GitHub
    gh_pat: str = field(default_factory=lambda: os.environ.get("GH_PAT", ""))
    github_username: str = field(
        default_factory=lambda: os.environ.get("GITHUB_USERNAME", "atharvadevne123")
    )

    # Notion
    notion_api_key: str = field(
        default_factory=lambda: os.environ.get("NOTION_API_KEY", "")
    )
    notion_database_id: str = field(
        default_factory=lambda: os.environ.get("NOTION_DATABASE_ID", "")
    )

    # Gmail
    gmail_user: str = field(default_factory=lambda: os.environ.get("GMAIL_USER", ""))
    gmail_app_pass: str = field(
        default_factory=lambda: os.environ.get("GMAIL_APP_PASS", "")
    )
    report_recipient: str = field(
        default_factory=lambda: os.environ.get(
            "REPORT_RECIPIENT", os.environ.get("GMAIL_USER", "")
        )
    )

    # Logging
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO").upper()
    )
    json_logs: bool = field(
        default_factory=lambda: os.environ.get("JSON_LOGS", "0") == "1"
    )

    # Report settings
    report_subject_prefix: str = field(
        default_factory=lambda: os.environ.get(
            "REPORT_SUBJECT_PREFIX", "[Reflective Lantern]"
        )
    )
    dry_run: bool = field(
        default_factory=lambda: os.environ.get("DRY_RUN", "0") == "1"
    )

    # Run settings
    pf_fix_timeout: int = field(
        default_factory=lambda: int(os.environ.get("PF_FIX_TIMEOUT", "300"))
    )
    commit_target: int = field(
        default_factory=lambda: int(os.environ.get("COMMIT_TARGET", "60"))
    )

    def validate(self) -> list[str]:
        """Return a list of missing required environment variable names.

        An empty list means all required credentials are present and non-empty.
        """
        missing: list[str] = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.gh_pat:
            missing.append("GH_PAT")
        if not self.notion_api_key:
            missing.append("NOTION_API_KEY")
        if not self.gmail_user:
            missing.append("GMAIL_USER")
        if not self.gmail_app_pass:
            missing.append("GMAIL_APP_PASS")
        return missing

    def is_valid(self) -> bool:
        """Return True when all required credentials are present."""
        return len(self.validate()) == 0

    def __repr__(self) -> str:
        """Return repr with sensitive fields masked."""
        _SENSITIVE = {"anthropic_api_key", "gh_pat", "notion_api_key", "gmail_app_pass"}
        parts = []
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if f in _SENSITIVE:
                val = "***" if val else ""
            parts.append(f"{f}={val!r}")
        return f"Settings({', '.join(parts)})"

    @property
    def history_dir(self) -> Path:
        """Absolute path to the history directory."""
        return Path(__file__).parent.parent / "history"


# Module-level singleton for convenience imports
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the module-level Settings singleton, creating it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
