"""Project-wide constants for Reflective Lantern."""

from __future__ import annotations

from pathlib import Path

# --- Versioning ---
VERSION: str = "1.0.0"

# --- Paths ---
ROOT_DIR: Path = Path(__file__).parent.parent
HISTORY_DIR: Path = ROOT_DIR / "history"
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"

# --- Commit / mode config ---
COMMIT_TARGET: int = 60
INNOVATION_WEEKDAY: int = 3  # Wednesday
INNOVATION_DAY_RANGES: list[tuple[int, int]] = [(8, 14), (22, 28)]

# --- History / housekeeping ---
NON_RECORD_FILES: set[str] = {
    "schema.json",
    "commit_schedule.json",
    "email_status.json",
    "pending",
}
MAX_HISTORY_ENTRIES: int = 500
CLEANUP_DEFAULT_DAYS: int = 90
WEEKLY_SUMMARY_DAYS: int = 7
MIN_REPOS_FOR_ROTATION: int = 1

# --- GitHub ---
GITHUB_OWNER: str = "atharvadevne123"
GITHUB_API_BASE: str = "https://api.github.com"
MAX_REPOS_PER_PAGE: int = 100

# --- Email / SMTP ---
SMTP_HOST: str = "smtp.gmail.com"
SMTP_PORT_TLS: int = 587
SMTP_PORT_SSL: int = 465
REPORT_DATE_FORMAT: str = "%Y-%m-%d"

# --- PDF ---
PDF_MAX_SIZE_MB: float = 10.0

# --- Notion ---
NOTION_MODEL: str = "claude-sonnet-4-6"

# --- Reporting ---
SEPARATOR_WIDTH: int = 80
DAILY_REPORT_HEADER: str = "Daily Report"
WEEKLY_REPORT_HEADER: str = "Weekly Summary"

__all__ = [
    "VERSION",
    "ROOT_DIR",
    "HISTORY_DIR",
    "SCRIPTS_DIR",
    "COMMIT_TARGET",
    "INNOVATION_WEEKDAY",
    "INNOVATION_DAY_RANGES",
    "NON_RECORD_FILES",
    "MAX_HISTORY_ENTRIES",
    "CLEANUP_DEFAULT_DAYS",
    "WEEKLY_SUMMARY_DAYS",
    "MIN_REPOS_FOR_ROTATION",
    "GITHUB_OWNER",
    "GITHUB_API_BASE",
    "MAX_REPOS_PER_PAGE",
    "SMTP_HOST",
    "SMTP_PORT_TLS",
    "SMTP_PORT_SSL",
    "REPORT_DATE_FORMAT",
    "PDF_MAX_SIZE_MB",
    "NOTION_MODEL",
    "SEPARATOR_WIDTH",
    "DAILY_REPORT_HEADER",
    "WEEKLY_REPORT_HEADER",
]
