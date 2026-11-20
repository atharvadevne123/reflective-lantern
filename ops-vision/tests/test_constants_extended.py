"""Extended tests for ops-vision app/constants.py helpers and constants."""

from __future__ import annotations

import pytest

from app.constants import (
    HEALTH_EXEMPT_PATHS,
    MAX_BATCH_SIZE,
    MAX_TOP_K,
    SEVERITY_LEVELS,
    is_valid_severity,
    severity_from_confidence,
)


class TestSeverityLevels:
    def test_contains_all_four_levels(self) -> None:
        for level in ("low", "medium", "high", "critical"):
            assert level in SEVERITY_LEVELS

    def test_is_tuple(self) -> None:
        assert isinstance(SEVERITY_LEVELS, tuple)


class TestIsValidSeverity:
    @pytest.mark.parametrize("label", ["low", "medium", "high", "critical"])
    def test_valid_labels(self, label: str) -> None:
        assert is_valid_severity(label) is True

    def test_invalid_label_false(self) -> None:
        assert is_valid_severity("extreme") is False

    def test_empty_string_false(self) -> None:
        assert is_valid_severity("") is False

    def test_case_sensitive(self) -> None:
        assert is_valid_severity("LOW") is False
        assert is_valid_severity("CRITICAL") is False


class TestSeverityFromConfidence:
    def test_critical_at_high_confidence(self) -> None:
        assert severity_from_confidence(0.95) == "critical"

    def test_critical_at_exact_threshold(self) -> None:
        assert severity_from_confidence(0.9) == "critical"

    def test_high_in_range(self) -> None:
        assert severity_from_confidence(0.80) == "high"

    def test_medium_in_range(self) -> None:
        assert severity_from_confidence(0.60) == "medium"

    def test_low_at_zero(self) -> None:
        assert severity_from_confidence(0.0) == "low"

    def test_low_below_medium_threshold(self) -> None:
        assert severity_from_confidence(0.4) == "low"

    @pytest.mark.parametrize("conf,expected", [
        (1.0, "critical"),
        (0.9, "critical"),
        (0.75, "high"),
        (0.5, "medium"),
        (0.0, "low"),
    ])
    def test_boundary_values(self, conf: float, expected: str) -> None:
        assert severity_from_confidence(conf) == expected


class TestConstants:
    def test_max_batch_size_positive(self) -> None:
        assert MAX_BATCH_SIZE > 0

    def test_max_top_k_positive(self) -> None:
        assert MAX_TOP_K > 0

    def test_health_exempt_paths_frozenset(self) -> None:
        assert isinstance(HEALTH_EXEMPT_PATHS, frozenset)
        assert "/health" in HEALTH_EXEMPT_PATHS
