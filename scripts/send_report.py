"""Send Reflective Lantern daily report email with PDF attachment."""
from __future__ import annotations

import smtplib
import ssl
import sys
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

DATE = "2026-08-14"
RECIPIENT = "devneatharva@gmail.com"
SENDER = "devneatharva@gmail.com"
APP_PASS = "mzizhaaiaemssxls"
PDF_PATH = Path("/tmp/lantern-work/reflective-lantern/history/reports/reflective-lantern_2026-08-14.pdf")

BODY = f"""\
Reflective Lantern — Daily Report ({DATE})

Mode:    IMPROVEMENT
Commits: 61 (60 improvements + 1 ruff cleanup)
Branch:  main
Ruff:    all checks passed

Highlights:
  • Expanded test parametrization across 30 test files
  • Added 100+ new parametrized tests in main app and all 5 sub-projects
  • Fixed 4 linting/syntax bugs (F811 duplicate imports, exception chaining)
  • Sub-projects improved: energy_seer, temporal-pulse, price-prophet, suite-cast, forge-guard

See the attached PDF for the full commit breakdown.

-- Reflective Lantern Bot
"""

def send_email() -> str:
    msg = MIMEMultipart()
    msg["Subject"] = f"[Reflective Lantern] IMPROVEMENT Report — {DATE}"
    msg["From"] = SENDER
    msg["To"] = RECIPIENT
    msg.attach(MIMEText(BODY, "plain"))

    if PDF_PATH.exists():
        with open(PDF_PATH, "rb") as f:
            part = MIMEApplication(f.read(), Name=PDF_PATH.name)
        part["Content-Disposition"] = f'attachment; filename="{PDF_PATH.name}"'
        msg.attach(part)

    # Try port 587 with STARTTLS
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ctx)
            smtp.login(SENDER, APP_PASS)
            smtp.sendmail(SENDER, [RECIPIENT], msg.as_string())
        return "sent_587"
    except Exception:
        pass

    # Try port 465 with SSL
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=15) as smtp:
            smtp.login(SENDER, APP_PASS)
            smtp.sendmail(SENDER, [RECIPIENT], msg.as_string())
        return "sent_465"
    except Exception as e465:
        return f"smtp_blocked: {type(e465).__name__}: {e465}"


if __name__ == "__main__":
    status = send_email()
    print(status)
    sys.exit(0)
