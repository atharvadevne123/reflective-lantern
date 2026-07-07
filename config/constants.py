"""Project-wide constants for Reflective Lantern."""

from __future__ import annotations

from pathlib import Path

VERSION: str = "1.0.0"

# Repository root
ROOT_DIR: Path = Path(__file__).parent.parent

# Directory paths
HISTORY_DIR: Path = ROOT_DIR / "history"
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"
COVERS_DIR: Path = ROOT_DIR / "covers"
PROMPTS_DIR: Path = ROOT_DIR / "prompts"
DOCS_DIR: Path = ROOT_DIR / "docs"

# Run configuration
COMMIT_TARGET: int = 60
PF_FIX_TIMEOUT_SECONDS: int = 300
MAX_REPOS_PER_PAGE: int = 100

# Innovation mode: fires on Wednesday days 8-14 or 22-28
INNOVATION_WEEKDAY: int = 3  # isoweekday Wednesday
INNOVATION_DAY_RANGES: tuple[tuple[int, int], ...] = ((8, 14), (22, 28))

# Email / SMTP
SMTP_HOST: str = "smtp.gmail.com"
SMTP_PORT_TLS: int = 587
SMTP_PORT_SSL: int = 465

# GitHub API
GITHUB_API_BASE: str = "https://api.github.com"
GITHUB_OWNER: str = "atharvadevne123"

# Notion covers base URL
COVER_BASE_URL: str = (
    "https://raw.githubusercontent.com/atharvadevne123/reflective-lantern/main/covers"
)

# Claude model for portfolio descriptions
NOTION_MODEL: str = "claude-sonnet-4-6"
NOTION_DESCRIPTION_MAX_TOKENS: int = 200

# History files that are config/meta files, not per-run records
NON_RECORD_FILES: frozenset[str] = frozenset({"schema.json", "commit_schedule.json"})

# Cleanup defaults
CLEANUP_DEFAULT_DAYS: int = 90
MAX_HISTORY_ENTRIES: int = 365

# Report formatting
REPORT_DATE_FORMAT: str = "%Y-%m-%d"
PDF_MAX_SIZE_MB: float = 5.0

# Weekly summary window
WEEKLY_SUMMARY_DAYS: int = 7

# Repo selection constraints
MIN_REPOS_FOR_ROTATION: int = 1
