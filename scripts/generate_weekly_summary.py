#!/usr/bin/env python3
"""Generate and email a weekly Reflective Lantern summary.

Reads all history JSON files, aggregates the past 7 days of runs,
builds a PDF + plain text body, and sends via Gmail.

Usage:
    python scripts/generate_weekly_summary.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


def _build_body(end_date: date) -> str:
    """Import report_generator to build the weekly Markdown body."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.report_generator import weekly_report

    return weekly_report(end_date)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Send weekly Reflective Lantern summary email")
    parser.add_argument("--dry-run", action="store_true", help="Print email body without sending")
    parser.add_argument("--subject", default=None, help="Override the email subject line")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    end_date = date.today()
    start_date = end_date - timedelta(days=6)
    body = _build_body(end_date)

    subject = args.subject or (
        f"Reflective Lantern Weekly Summary — {start_date.isoformat()} to {end_date.isoformat()}"
    )

    if args.dry_run:
        print(f"Subject: {subject}")
        print()
        print(body)
        return

    from scripts.email_report import send_report

    sent = send_report(subject=subject, body=body)
    if not sent:
        log.error("Failed to send weekly summary email")


if __name__ == "__main__":
    main()
