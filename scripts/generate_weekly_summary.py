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
import os
import smtplib
import ssl
from datetime import date, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

log = logging.getLogger(__name__)


def _build_body(end_date: date) -> str:
    """Import report_generator to build the weekly Markdown body."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.report_generator import weekly_report
    return weekly_report(end_date)


def send_email(
    subject: str,
    body: str,
    sender: str,
    password: str,
    recipient: str,
    pdf_path: Path | None = None,
) -> None:
    """Send *body* as a plain-text email, optionally attaching *pdf_path*."""
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(body, "plain"))

    if pdf_path and pdf_path.exists():
        with pdf_path.open("rb") as fh:
            att = MIMEApplication(fh.read(), _subtype="pdf")
        att.add_header("Content-Disposition", "attachment", filename=pdf_path.name)
        msg.attach(att)

    ctx = ssl.create_default_context()
    for port, use_ssl in [(587, False), (465, True)]:
        try:
            if use_ssl:
                with smtplib.SMTP_SSL("smtp.gmail.com", port, timeout=20, context=ctx) as s:
                    s.login(sender, password)
                    s.sendmail(sender, recipient, msg.as_string())
            else:
                with smtplib.SMTP("smtp.gmail.com", port, timeout=20) as s:
                    s.ehlo()
                    s.starttls(context=ctx)
                    s.ehlo()
                    s.login(sender, password)
                    s.sendmail(sender, recipient, msg.as_string())
            log.info("Email sent via port %d", port)
            return
        except Exception as exc:
            log.warning("Port %d failed: %s", port, exc)
    raise RuntimeError("All SMTP ports failed")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Send weekly Reflective Lantern summary email")
    parser.add_argument("--dry-run", action="store_true", help="Print email body without sending")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    end_date = date.today()
    start_date = end_date - timedelta(days=6)
    body = _build_body(end_date)

    subject = (
        f"Reflective Lantern Weekly Summary — "
        f"{start_date.isoformat()} to {end_date.isoformat()}"
    )

    if args.dry_run:
        print(f"Subject: {subject}")
        print()
        print(body)
        return

    sender = os.environ.get("GMAIL_USER", "")
    password = os.environ.get("GMAIL_APP_PASS", "")
    recipient = os.environ.get("REPORT_RECIPIENT", sender)

    if not sender or not password:
        log.error("GMAIL_USER and GMAIL_APP_PASS must be set")
        return

    send_email(subject, body, sender, password, recipient)


if __name__ == "__main__":
    main()
