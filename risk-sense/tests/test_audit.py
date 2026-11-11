"""Tests for the audit logging module."""

from __future__ import annotations

import json

import pytest

from app.audit import (
    _hash_applicant_id,
    build_audit_record,
    format_audit_record_json,
    log_decision,
)


class TestHashApplicantId:
    def test_returns_16_char_string(self) -> None:
        h = _hash_applicant_id("app-001")
        assert len(h) == 16

    def test_same_id_same_hash(self) -> None:
        assert _hash_applicant_id("app-001") == _hash_applicant_id("app-001")

    def test_different_ids_different_hashes(self) -> None:
        assert _hash_applicant_id("app-001") != _hash_applicant_id("app-002")

    def test_original_id_not_in_hash(self) -> None:
        applicant_id = "sensitive-applicant-id"
        h = _hash_applicant_id(applicant_id)
        assert applicant_id not in h


class TestBuildAuditRecord:
    def test_contains_required_fields(self) -> None:
        record = build_audit_record(
            applicant_id="app-001",
            risk_score=0.7,
            risk_category="high",
            model_version="1.0.0",
            threshold_used=0.5,
            decision="denied",
        )
        for field in ("applicant_hash", "risk_score", "risk_category", "decision",
                      "threshold_used", "model_version", "decided_at"):
            assert field in record

    def test_decision_is_denied_for_high_score(self) -> None:
        record = build_audit_record(
            applicant_id="app-002",
            risk_score=0.8,
            risk_category="very_high",
            model_version="1.0.0",
            threshold_used=0.5,
            decision="denied",
        )
        assert record["decision"] == "denied"

    def test_adverse_reasons_default_empty(self) -> None:
        record = build_audit_record(
            applicant_id="app-003",
            risk_score=0.3,
            risk_category="low",
            model_version="1.0.0",
            threshold_used=0.5,
            decision="approved",
        )
        assert record["adverse_reasons"] == []

    def test_original_id_not_exposed(self) -> None:
        applicant_id = "top-secret-id"
        record = build_audit_record(
            applicant_id=applicant_id,
            risk_score=0.4,
            risk_category="medium",
            model_version="1.0.0",
            threshold_used=0.5,
            decision="approved",
        )
        assert applicant_id not in str(record)


class TestFormatAuditRecordJson:
    def test_valid_json(self) -> None:
        record = {"key": "value", "score": 0.5}
        json_str = format_audit_record_json(record)
        parsed = json.loads(json_str)
        assert parsed == record

    def test_compact_no_spaces(self) -> None:
        record = {"a": 1, "b": 2}
        json_str = format_audit_record_json(record)
        assert " " not in json_str


class TestLogDecision:
    def test_approved_for_low_score(self) -> None:
        record = log_decision("app-004", risk_score=0.3, risk_category="low")
        assert record["decision"] == "approved"

    def test_denied_for_high_score(self) -> None:
        record = log_decision("app-005", risk_score=0.8, risk_category="high")
        assert record["decision"] == "denied"

    def test_custom_threshold(self) -> None:
        record = log_decision("app-006", risk_score=0.4, risk_category="medium", threshold_used=0.3)
        assert record["decision"] == "denied"
