"""Audit logging for credit risk decisions (regulatory compliance)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _hash_applicant_id(applicant_id: str) -> str:
    """One-way hash of applicant_id for audit log storage without PII."""
    return hashlib.sha256(applicant_id.encode()).hexdigest()[:16]


def build_audit_record(
    applicant_id: str,
    risk_score: float,
    risk_category: str,
    model_version: str,
    threshold_used: float,
    decision: str,
    adverse_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build a structured audit record for a credit decision.

    The applicant_id is hashed so the audit log does not retain PII;
    the original ID lives in the application database keyed by the hash.
    """
    return {
        "applicant_hash": _hash_applicant_id(applicant_id),
        "risk_score": round(risk_score, 6),
        "risk_category": risk_category,
        "decision": decision,
        "threshold_used": round(threshold_used, 4),
        "model_version": model_version,
        "adverse_reasons": adverse_reasons or [],
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }


def format_audit_record_json(record: dict[str, Any]) -> str:
    """Serialise audit record to compact JSON for structured log output."""
    return json.dumps(record, separators=(",", ":"))


def log_decision(
    applicant_id: str,
    risk_score: float,
    risk_category: str,
    model_version: str = "1.0.0",
    threshold_used: float = 0.5,
    adverse_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Log a credit decision to the structured audit logger and return the record.

    Uses logger.info so the record appears in any configured handler
    (stdout JSON lines, file sink, cloud log aggregator).
    """
    decision = "denied" if risk_score >= threshold_used else "approved"
    record = build_audit_record(
        applicant_id=applicant_id,
        risk_score=risk_score,
        risk_category=risk_category,
        model_version=model_version,
        threshold_used=threshold_used,
        decision=decision,
        adverse_reasons=adverse_reasons,
    )
    logger.info("AUDIT %s", format_audit_record_json(record))
    return record
