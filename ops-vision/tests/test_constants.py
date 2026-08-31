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
)


class TestSeverityConstants:
    """Tests that severity threshold constants are ordered correctly."""

    def test_medium_lt_high(self):
        """MEDIUM threshold is below HIGH."""
        assert SEVERITY_MEDIUM_THRESHOLD < SEVERITY_HIGH_THRESHOLD

    def test_high_lt_critical(self):
        """HIGH threshold is below CRITICAL."""
        assert SEVERITY_HIGH_THRESHOLD < SEVERITY_CRITICAL_THRESHOLD

    def test_critical_threshold_le_one(self):
        """CRITICAL threshold is at most 1.0."""
        assert SEVERITY_CRITICAL_THRESHOLD <= 1.0

    def test_medium_threshold_ge_zero(self):
        """MEDIUM threshold is at least 0.0."""
        assert SEVERITY_MEDIUM_THRESHOLD >= 0.0

    def test_severity_levels_contains_all_four(self):
        """SEVERITY_LEVELS contains low, medium, high, critical."""
        assert set(SEVERITY_LEVELS) == {"low", "medium", "high", "critical"}


class TestInferSeverity:
    """Tests for the _infer_severity helper in routes."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from app.api.v1.routes import _infer_severity
        self._infer = _infer_severity

    def test_critical_at_threshold(self):
        """Confidence at or above 0.9 maps to critical."""
        assert self._infer(0.9) == "critical"
        assert self._infer(1.0) == "critical"

    def test_high_at_threshold(self):
        """Confidence at 0.75 maps to high."""
        assert self._infer(0.75) == "high"

    def test_medium_at_threshold(self):
        """Confidence at 0.5 maps to medium."""
        assert self._infer(0.5) == "medium"

    def test_low_below_medium_threshold(self):
        """Confidence below 0.5 maps to low."""
        assert self._infer(0.49) == "low"
        assert self._infer(0.0) == "low"

    @pytest.mark.parametrize("conf,expected", [
        (0.95, "critical"),
        (0.80, "high"),
        (0.60, "medium"),
        (0.30, "low"),
    ])
    def test_severity_levels_parametrized(self, conf, expected):
        """Parametrized boundary checks for all severity levels."""
        assert self._infer(conf) == expected


class TestOtherConstants:
    """Tests for limit and path constants."""

    def test_max_batch_size_positive(self):
        """MAX_BATCH_SIZE is a positive integer."""
        assert MAX_BATCH_SIZE > 0

    def test_max_top_k_positive(self):
        """MAX_TOP_K is a positive integer."""
        assert MAX_TOP_K > 0

    def test_min_query_length_positive(self):
        """MIN_QUERY_LENGTH is a positive integer."""
        assert MIN_QUERY_LENGTH > 0

    def test_max_pagination_gt_max_batch(self):
        """MAX_PAGINATION_LIMIT is larger than MAX_BATCH_SIZE."""
        assert MAX_PAGINATION_LIMIT > MAX_BATCH_SIZE

    def test_health_exempt_paths_is_frozenset(self):
        """HEALTH_EXEMPT_PATHS is a frozenset."""
        assert isinstance(HEALTH_EXEMPT_PATHS, frozenset)

    def test_health_path_is_exempt(self):
        """'/health' is in HEALTH_EXEMPT_PATHS."""
        assert "/health" in HEALTH_EXEMPT_PATHS
