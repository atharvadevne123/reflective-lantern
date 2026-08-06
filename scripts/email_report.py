"""Email report sender for Reflective Lantern."""

from __future__ import annotations

import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from config.constants import SMTP_HOST, SMTP_PORT_SSL, SMTP_PORT_TLS


def build_message(
    subject: str,
    body: str,
    from_addr: str,
    to_addr: str,
    *,
    pdf_path: Path | None = None,
    subject_prefix: str = "",
) -> MIMEMultipart:
    """Build a MIME email message with optional PDF attachment."""
    full_subject = f"{subject_prefix} {subject}".strip() if subject_prefix else subject

    msg = MIMEMultipart()
    msg["Subject"] = full_subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(body, "plain"))

    if pdf_path is not None and pdf_path.exists():
        with pdf_path.open("rb") as fh:
            part = MIMEApplication(fh.read(), _subtype="pdf")
            part.add_header("Content-Disposition", "attachment", filename=pdf_path.name)
            msg.attach(part)

    return msg


def send_via_smtp(
    msg: MIMEMultipart,
    user: str,
    password: str,
    to_addr: str,
) -> None:
    """Send *msg* via Gmail SMTP (tries TLS then SSL)."""
    errors = []
    for use_ssl, port in [(False, SMTP_PORT_TLS), (True, SMTP_PORT_SSL)]:
        try:
            if use_ssl:
                with smtplib.SMTP_SSL(SMTP_HOST, port) as server:
                    server.login(user, password)
                    server.sendmail(user, to_addr, msg.as_string())
            else:
                with smtplib.SMTP(SMTP_HOST, port) as server:
                    server.ehlo()
                    server.starttls()
                    server.login(user, password)
                    server.sendmail(user, to_addr, msg.as_string())
            return
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError(f"All SMTP attempts failed: {'; '.join(errors)}")


def send_report(
    subject: str,
    body: str,
    *,
    pdf_path: Path | None = None,
) -> bool:
    """Send an email report; returns True on success, False if credentials missing or send fails."""
    user = os.environ.get("GMAIL_USER", "")
    password = os.environ.get("GMAIL_APP_PASS", "")
    recipient = os.environ.get("REPORT_RECIPIENT", user)

    if not user or not password:
        return False

    try:
        msg = build_message(subject, body, user, recipient, pdf_path=pdf_path)
        send_via_smtp(msg, user, password, recipient)
        return True
    except Exception:
        return False


__all__ = ["build_message", "send_report", "send_via_smtp"]
