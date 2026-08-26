"""Deliver a Reflective Lantern run report.

Delivery adapts to the execution context rather than assuming it. When SMTP is
reachable the report is emailed; when it is not — as in any sandbox whose
egress is restricted to an HTTPS proxy — the report is written into the
repository under ``reports/`` so the run still produces a durable artifact
instead of failing.

Credentials are read from the environment, never hardcoded:

    LANTERN_SMTP_USER      sender address        (default: LANTERN_REPORT_TO)
    LANTERN_SMTP_PASSWORD  SMTP app password     (required to send email)
    LANTERN_REPORT_TO      recipient address
    LANTERN_SMTP_HOST      SMTP host             (default: smtp.gmail.com)

If ``LANTERN_SMTP_PASSWORD`` is unset, email is skipped and the report is
filed to disk. That is a normal outcome, not an error.

Usage:
    python3 scripts/send_report.py --subject "..." --body-file body.txt \\
        [--attach report.pdf]
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
import ssl
import sys
from datetime import date
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
SMTP_PORTS: tuple[tuple[int, bool], ...] = ((587, False), (465, True))


def _build_message(subject: str, body: str, sender: str, recipient: str,
                   attachment: Path | None) -> MIMEMultipart:
    """Assemble the MIME message.

    Args:
        subject: Email subject line.
        body: Plain-text body.
        sender: From address.
        recipient: To address.
        attachment: Optional file to attach.

    Returns:
        A MIMEMultipart message ready to send.
    """
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(body, "plain"))

    if attachment and attachment.exists():
        subtype = attachment.suffix.lstrip(".") or "octet-stream"
        part = MIMEApplication(attachment.read_bytes(), _subtype=subtype)
        part.add_header(
            "Content-Disposition", "attachment", filename=attachment.name
        )
        msg.attach(part)

    return msg


def send_email(subject: str, body: str, attachment: Path | None = None) -> bool:
    """Attempt to deliver the report by email.

    Args:
        subject: Email subject line.
        body: Plain-text body.
        attachment: Optional file to attach.

    Returns:
        True if the message was accepted by an SMTP server, else False.
    """
    password = os.environ.get("LANTERN_SMTP_PASSWORD")
    recipient = os.environ.get("LANTERN_REPORT_TO")
    sender = os.environ.get("LANTERN_SMTP_USER", recipient or "")
    host = os.environ.get("LANTERN_SMTP_HOST", "smtp.gmail.com")

    if not password or not recipient:
        logger.info(
            "LANTERN_SMTP_PASSWORD/LANTERN_REPORT_TO not set — skipping email"
        )
        return False

    msg = _build_message(subject, body, sender, recipient, attachment)

    for port, use_ssl in SMTP_PORTS:
        try:
            if use_ssl:
                with smtplib.SMTP_SSL(host, port, timeout=20) as server:
                    server.login(sender, password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(host, port, timeout=20) as server:
                    server.ehlo()
                    server.starttls(context=ssl.create_default_context())
                    server.ehlo()
                    server.login(sender, password)
                    server.send_message(msg)
            logger.info("Report emailed via %s:%s", host, port)
            return True
        except Exception as exc:  # noqa: BLE001 - any failure means try next port
            logger.warning("SMTP %s:%s failed: %s", host, port, exc)

    return False


def file_to_repo(subject: str, body: str, attachment: Path | None = None) -> Path:
    """Write the report into the repository as a durable fallback.

    Args:
        subject: Used as the report heading.
        body: Report text.
        attachment: Optional file to copy alongside the text report.

    Returns:
        Path to the written text report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in subject)[:60]
    path = REPORTS_DIR / f"{stamp}_{slug}.txt".replace("--", "-")

    path.write_text(f"{subject}\n{'=' * len(subject)}\n\n{body}\n", encoding="utf-8")
    logger.info("Report filed to %s", path)

    if attachment and attachment.exists() and attachment.parent != REPORTS_DIR:
        target = REPORTS_DIR / attachment.name
        target.write_bytes(attachment.read_bytes())
        logger.info("Attachment copied to %s", target)

    return path


def deliver(subject: str, body: str, attachment: Path | None = None) -> str:
    """Deliver the report by the best channel available.

    Email is attempted first; if it is unavailable for any reason the report is
    filed into the repository instead. Delivery never raises — a run should not
    fail because its reporting channel is down.

    Args:
        subject: Report subject.
        body: Report text.
        attachment: Optional file to include.

    Returns:
        A status string describing how the report was delivered.
    """
    try:
        if send_email(subject, body, attachment):
            return "emailed"
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error while emailing — falling back to file")

    path = file_to_repo(subject, body, attachment)
    return f"filed:{path.relative_to(REPORTS_DIR.parent)}"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Optional argument vector, defaulting to sys.argv[1:].

    Returns:
        Process exit code (always 0 — delivery failure is not a run failure).
    """
    parser = argparse.ArgumentParser(description="Deliver a Reflective Lantern report")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body-file", type=Path, required=True)
    parser.add_argument("--attach", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    body = args.body_file.read_text(encoding="utf-8")
    status = deliver(args.subject, body, args.attach)
    print(f"Delivery: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
