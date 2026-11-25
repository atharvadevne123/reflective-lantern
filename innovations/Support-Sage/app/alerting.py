"""Drift alerting via logging and optional webhook."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def alert_if_drift(drift_report: dict[str, Any], webhook_url: str | None = None) -> None:
    """Log and optionally POST a webhook when drift is detected."""
    if not drift_report.get("any_drift_detected"):
        return
    msg = f"Support-Sage: drift detected. KS results: {drift_report.get('drift')}"
    logger.warning(msg)
    if webhook_url:
        try:
            import urllib.request
            data = f'{{"text": "{msg}"}}'.encode()
            req = urllib.request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            logger.warning("Drift webhook failed: %s", exc)
