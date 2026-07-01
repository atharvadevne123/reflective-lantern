#!/usr/bin/env python3
"""Reusable email utility for sending Reflective Lantern PDF reports.

Usage:
    from scripts.email_report import send_report
    send_report(
        subject="Reflective Lantern: MyRepo (60 commits) - 2026-07-01",
        body="See attached PDF for details.",
        pdf_path=Path("/tmp/report.pdf"),
    )
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SMTP_HOST = "smtp.gmail.com"
SMTP_ATTEMPTS = [(587, False), (465, True)]


def build_message(
    subject: str,
    body: str,
    sender: str,
    recipient: str,
    pdf_path: Optional[Path] = None,
) -> MIMEMultipart:
    """Build a MIME multipart email, optionally attaching a PDF."""
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(body, "plain"))
    if pdf_path and pdf_path.is_file():
        with pdf_path.open("rb") as fh:
            att = MIMEApplication(fh.read(), _subtype="pdf")
        att.add_header("Content-Disposition", "attachment", filename=pdf_path.name)
        msg.attach(att)
    return msg


def send_via_smtp(
    msg: MIMEMultipart,
    sender: str,
    password: str,
    recipient: str,
) -> None:
    """Attempt to send *msg* via SMTP, trying TLS then SSL ports.

    Raises:
        RuntimeError: If all SMTP ports fail.
    """
    ctx = ssl.create_default_context()
    last_exc: Exception | None = None
    for port, use_ssl in SMTP_ATTEMPTS:
        try:
            if use_ssl:
                with smtplib.SMTP_SSL(SMTP_HOST, port, timeout=20, context=ctx) as s:
                    s.login(sender, password)
                    s.sendmail(sender, recipient, msg.as_string())
            else:
                with smtplib.SMTP(SMTP_HOST, port, timeout=20) as s:
                    s.ehlo()
                    s.starttls(context=ctx)
                    s.ehlo()
                    s.login(sender, password)
                    s.sendmail(sender, recipient, msg.as_string())
            log.info("Email sent via %s port %d", SMTP_HOST, port)
            return
        except Exception as exc:
            log.warning("SMTP port %d failed: %s", port, exc)
            last_exc = exc
    raise RuntimeError(f"All SMTP ports failed. Last error: {last_exc}") from last_exc


def send_report(
    subject: str,
    body: str,
    pdf_path: Optional[Path] = None,
    sender: Optional[str] = None,
    password: Optional[str] = None,
    recipient: Optional[str] = None,
) -> bool:
    """Build and send a report email. Returns True on success.

    Falls back to GMAIL_USER, GMAIL_APP_PASS, and REPORT_RECIPIENT env vars
    if explicit values are not provided.
    """
    _sender = sender or os.environ.get("GMAIL_USER", "")
    _password = password or os.environ.get("GMAIL_APP_PASS", "")
    _recipient = recipient or os.environ.get("REPORT_RECIPIENT", _sender)

    if not _sender or not _password:
        log.error("GMAIL_USER and GMAIL_APP_PASS must be set to send email")
        return False

    msg = build_message(subject, body, _sender, _recipient, pdf_path)
    try:
        send_via_smtp(msg, _sender, _password, _recipient)
        return True
    except RuntimeError as exc:
        log.error("Failed to send email: %s", exc)
        return False
