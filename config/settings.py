"""Runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar


class Settings:
    """Frozen settings object populated from environment variables."""

    _REQUIRED: ClassVar[list[str]] = ["ANTHROPIC_API_KEY", "GH_PAT", "NOTION_API_KEY"]
    _SENSITIVE: ClassVar[set[str]] = {
        "anthropic_api_key",
        "gh_pat",
        "notion_api_key",
        "gmail_app_pass",
        "foundry_token",
    }

    __slots__ = (
        "anthropic_api_key",
        "gh_pat",
        "notion_api_key",
        "gmail_user",
        "gmail_app_pass",
        "report_recipient",
        "report_subject_prefix",
        "log_level",
        "json_logs",
        "commit_target",
        "pf_fix_timeout",
        "dry_run",
        "foundry_hostname",
        "foundry_token",
        "foundry_dataset_rid",
    )

    def __init__(self) -> None:
        object.__setattr__(self, "anthropic_api_key", os.getenv("ANTHROPIC_API_KEY", ""))
        object.__setattr__(self, "gh_pat", os.getenv("GH_PAT", ""))
        object.__setattr__(self, "notion_api_key", os.getenv("NOTION_API_KEY", ""))
        gmail_user = os.getenv("GMAIL_USER", "")
        object.__setattr__(self, "gmail_user", gmail_user)
        object.__setattr__(self, "gmail_app_pass", os.getenv("GMAIL_APP_PASS", ""))
        object.__setattr__(self, "report_recipient", os.getenv("REPORT_RECIPIENT", gmail_user))
        object.__setattr__(
            self,
            "report_subject_prefix",
            os.getenv("REPORT_SUBJECT_PREFIX", "[Reflective Lantern]"),
        )
        object.__setattr__(self, "log_level", os.getenv("LOG_LEVEL", "INFO"))
        object.__setattr__(self, "json_logs", os.getenv("JSON_LOGS", "0") in ("1", "true", "True"))
        object.__setattr__(self, "commit_target", int(os.getenv("COMMIT_TARGET", "60")))
        object.__setattr__(self, "pf_fix_timeout", int(os.getenv("PF_FIX_TIMEOUT", "300")))
        object.__setattr__(self, "dry_run", os.getenv("DRY_RUN", "0") in ("1", "true", "True"))
        object.__setattr__(self, "foundry_hostname", os.getenv("FOUNDRY_HOSTNAME", ""))
        object.__setattr__(self, "foundry_token", os.getenv("FOUNDRY_TOKEN", ""))
        object.__setattr__(self, "foundry_dataset_rid", os.getenv("FOUNDRY_DATASET_RID", ""))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Settings are frozen")

    def __repr__(self) -> str:
        from config.constants import GITHUB_OWNER

        parts = []
        for slot in self.__slots__:
            val = getattr(self, slot)
            if slot in self._SENSITIVE and val:
                parts.append(f"{slot}=***")
            else:
                parts.append(f"{slot}={val!r}")
        parts.append(f"github_owner={GITHUB_OWNER!r}")
        return f"Settings({', '.join(parts)})"

    def validate(self) -> list[str]:
        """Return a list of missing required env var names."""
        missing = []
        for key in self._REQUIRED:
            attr = key.lower()
            if not getattr(self, attr, ""):
                missing.append(key)
        return missing

    def is_valid(self) -> bool:
        """Return True when all required env vars are present."""
        return len(self.validate()) == 0

    def email_configured(self) -> bool:
        """Return True when GMAIL_USER and GMAIL_APP_PASS are both set."""
        return bool(self.gmail_user and self.gmail_app_pass)

    def foundry_configured(self) -> bool:
        """Return True when all three Foundry env vars are set."""
        return bool(self.foundry_hostname and self.foundry_token and self.foundry_dataset_rid)

    @property
    def history_dir(self) -> Path:
        """Return the project history directory path."""
        from config.constants import HISTORY_DIR
        return HISTORY_DIR

    @property
    def github_owner(self) -> str:
        """Return the GitHub owner/org from constants."""
        from config.constants import GITHUB_OWNER
        return GITHUB_OWNER


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the module-level Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
