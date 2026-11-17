"""Tests for veritas-rag trust scoring."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.scoring.trust import (
    SOURCE_QUALITY_PRIORS,
    freshness_score,
    source_quality_score,
)


def test_freshness_score_new_doc_near_one():
    now = datetime.now(UTC)
    created = now.isoformat()
    score = freshness_score(created, now=now)
    assert score > 0.99


def test_freshness_score_old_doc_less_than_half():
    now = datetime.now(UTC)
    created = (now - timedelta(days=730)).isoformat()
    score = freshness_score(created, half_life_days=365.0, now=now)
    assert score < 0.5


def test_freshness_score_exactly_half_life():
    now = datetime.now(UTC)
    created = (now - timedelta(days=365)).isoformat()
    score = freshness_score(created, half_life_days=365.0, now=now)
    assert abs(score - 0.5) < 0.01


def test_freshness_score_invalid_date_returns_neutral():
    score = freshness_score("not-a-date")
    assert score == 0.5


def test_freshness_score_in_range():
    now = datetime.now(UTC)
    created = (now - timedelta(days=100)).isoformat()
    score = freshness_score(created, now=now)
    assert 0.0 < score <= 1.0


def test_source_quality_pdf():
    score = source_quality_score("pdf")
    assert score == SOURCE_QUALITY_PRIORS["pdf"]


def test_source_quality_email():
    score = source_quality_score("email")
    assert score == SOURCE_QUALITY_PRIORS["email"]


def test_source_quality_pdf_higher_than_email():
    assert source_quality_score("pdf") > source_quality_score("email")


def test_source_quality_unknown_type_returns_default():
    score = source_quality_score("unknown_format")
    assert 0.0 < score <= 1.0


def test_source_quality_all_known_types_in_range():
    for source_type in SOURCE_QUALITY_PRIORS:
        score = source_quality_score(source_type)
        assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("source_type", ["pdf", "markdown", "text", "html", "email"])
def test_source_quality_all_types(source_type):
    score = source_quality_score(source_type)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
