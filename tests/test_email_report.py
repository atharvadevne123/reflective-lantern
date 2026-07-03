"""Tests for scripts.email_report."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_build_message_no_attachment() -> None:
    from scripts.email_report import build_message
    msg = build_message("Sub", "Body", "a@b.com", "c@d.com")
    assert msg["Subject"] == "Sub"
    assert msg["From"] == "a@b.com"
    assert msg["To"] == "c@d.com"


def test_build_message_with_pdf(tmp_path: Path) -> None:
    from scripts.email_report import build_message
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    msg = build_message("Sub", "Body", "a@b.com", "c@d.com", pdf_path=pdf)
    # Should have text part + pdf attachment = 2 parts
    parts = list(msg.walk())
    assert len(parts) >= 2


def test_build_message_missing_pdf_skipped(tmp_path: Path) -> None:
    from scripts.email_report import build_message
    missing = tmp_path / "missing.pdf"
    msg = build_message("Sub", "Body", "a@b.com", "c@d.com", pdf_path=missing)
    parts = [p for p in msg.walk() if p.get_content_type() == "application/pdf"]
    assert len(parts) == 0


def test_send_via_smtp_success() -> None:
    from email.mime.multipart import MIMEMultipart
    from scripts.email_report import send_via_smtp
    mock_smtp = MagicMock()
    mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp.__exit__ = MagicMock(return_value=False)
    with patch("smtplib.SMTP", return_value=mock_smtp):
        msg = MIMEMultipart()
        msg["Subject"] = "Test"
        send_via_smtp(msg, "a@b.com", "password", "c@d.com")
    mock_smtp.sendmail.assert_called_once()


def test_send_via_smtp_all_ports_fail() -> None:
    from email.mime.multipart import MIMEMultipart
    from scripts.email_report import send_via_smtp
    with (
        patch("smtplib.SMTP", side_effect=OSError("blocked")),
        patch("smtplib.SMTP_SSL", side_effect=OSError("blocked")),
        pytest.raises(RuntimeError, match="All"),
    ):
        msg = MIMEMultipart()
        send_via_smtp(msg, "a@b.com", "password", "c@d.com")


def test_send_report_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GMAIL_USER", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASS", raising=False)
    from scripts.email_report import send_report
    result = send_report(subject="S", body="B")
    assert result is False


@pytest.mark.parametrize("subject,body", [
    ("Daily Report", "60 commits pushed."),
    ("Weekly Summary", "7 repos updated this week."),
    ("Innovation: New-Repo", "Full ML project built from scratch."),
])
def test_build_message_various_subjects(subject: str, body: str) -> None:
    from scripts.email_report import build_message
    msg = build_message(subject, body, "sender@test.com", "recv@test.com")
    assert msg["Subject"] == subject
