"""Tests for shared constants and severity inference."""

import pytest

from app.constants import (
    HEALTH_EXEMPT_PATHS,
    MAX_BATCH_SIZE,
    MAX_PAGINATION_LIMIT,
    MAX_TOP_K,
    MIN_QUERY_LENGTH,
    SEVERITY_CRITICAL_THRESHOLD,
    SEVERITY_HIGH_THRESHOLD,
    SEVERITY_LEVELS,
    SEVERITY_MEDIUM_THRESHOLD,
    severity_from_confidence,
)


class TestSeverityConstants:
    """Tests that severity threshold constants are ordered correctly."""

    def test_medium_lt_high(self) -> None:
        """MEDIUM threshold is below HIGH."""
        assert SEVERITY_MEDIUM_THRESHOLD < SEVERITY_HIGH_THRESHOLD

    def test_high_lt_critical(self) -> None:
        """HIGH threshold is below CRITICAL."""
        assert SEVERITY_HIGH_THRESHOLD < SEVERITY_CRITICAL_THRESHOLD

    def test_critical_threshold_le_one(self) -> None:
        """CRITICAL threshold is at most 1.0."""
        assert SEVERITY_CRITICAL_THRESHOLD <= 1.0

    def test_medium_threshold_ge_zero(self) -> None:
        """MEDIUM threshold is at least 0.0."""
        assert SEVERITY_MEDIUM_THRESHOLD >= 0.0

    def test_severity_levels_contains_all_four(self) -> None:
        """SEVERITY_LEVELS contains low, medium, high, critical."""
        assert set(SEVERITY_LEVELS) == {"low", "medium", "high", "critical"}


class TestInferSeverity:
    """Tests for the _infer_severity helper in routes."""

    @pytest.fixture(autouse=True)
    def _import(self) -> None:
        from app.api.v1.routes import _infer_severity

        self._infer = _infer_severity

    def test_critical_at_threshold(self) -> None:
        """Confidence at or above 0.9 maps to critical."""
        assert self._infer(0.9) == "critical"
        assert self._infer(1.0) == "critical"

    def test_high_at_threshold(self) -> None:
        """Confidence at 0.75 maps to high."""
        assert self._infer(0.75) == "high"

    def test_medium_at_threshold(self) -> None:
        """Confidence at 0.5 maps to medium."""
        assert self._infer(0.5) == "medium"

    def test_low_below_medium_threshold(self) -> None:
        """Confidence below 0.5 maps to low."""
        assert self._infer(0.49) == "low"
        assert self._infer(0.0) == "low"

    @pytest.mark.parametrize(
        "conf,expected",
        [
            (0.95, "critical"),
            (0.80, "high"),
            (0.60, "medium"),
            (0.30, "low"),
        ],
    )
    def test_severity_levels_parametrized(self, conf, expected) -> None:
        """Parametrized boundary checks for all severity levels."""
        assert self._infer(conf) == expected


class TestOtherConstants:
    """Tests for limit and path constants."""

    def test_max_batch_size_positive(self) -> None:
        """MAX_BATCH_SIZE is a positive integer."""
        assert MAX_BATCH_SIZE > 0

    def test_max_top_k_positive(self) -> None:
        """MAX_TOP_K is a positive integer."""
        assert MAX_TOP_K > 0

    def test_min_query_length_positive(self) -> None:
        """MIN_QUERY_LENGTH is a positive integer."""
        assert MIN_QUERY_LENGTH > 0

    def test_max_pagination_gt_max_batch(self) -> None:
        """MAX_PAGINATION_LIMIT is larger than MAX_BATCH_SIZE."""
        assert MAX_PAGINATION_LIMIT > MAX_BATCH_SIZE

    def test_health_exempt_paths_is_frozenset(self) -> None:
        """HEALTH_EXEMPT_PATHS is a frozenset."""
        assert isinstance(HEALTH_EXEMPT_PATHS, frozenset)

    def test_health_path_is_exempt(self) -> None:
        """'/health' is in HEALTH_EXEMPT_PATHS."""
        assert "/health" in HEALTH_EXEMPT_PATHS


class TestSeverityFromConfidence:
    """Tests for the severity_from_confidence() helper."""

    @pytest.mark.parametrize(
        "confidence,expected",
        [
            (0.9, "critical"),
            (1.0, "critical"),
            (0.95, "critical"),
            (0.75, "high"),
            (0.85, "high"),
            (0.5, "medium"),
            (0.65, "medium"),
            (0.0, "low"),
            (0.1, "low"),
            (0.49, "low"),
        ],
    )
    def test_severity_mapping(self, confidence: float, expected: str) -> None:
        """severity_from_confidence maps each confidence range to the right level."""
        assert severity_from_confidence(confidence) == expected

    def test_returns_string(self) -> None:
        """severity_from_confidence always returns a str."""
        assert isinstance(severity_from_confidence(0.5), str)

    def test_result_is_in_severity_levels(self) -> None:
        """Result is always a known severity level."""
        for conf in [0.0, 0.3, 0.5, 0.75, 0.9, 1.0]:
            assert severity_from_confidence(conf) in SEVERITY_LEVELS
