"""Focused tests for the drift-detection helpers in app/monitoring.py.

These exercise ``compute_drift`` and ``drift_severity`` directly, without a
database, so the statistical behaviour is pinned independently of the ORM
aggregation paths covered in tests/test_monitoring.py.
"""

from __future__ import annotations

import random

import pytest

from app.monitoring import compute_drift, drift_severity

SAMPLE_SIZE = 200


@pytest.fixture
def rng() -> random.Random:
    """Seeded RNG so distribution tests are reproducible."""
    return random.Random(20260827)


class TestComputeDrift:
    def test_identical_distributions_show_no_drift(self, rng: random.Random) -> None:
        values = [rng.gauss(0.5, 0.1) for _ in range(SAMPLE_SIZE)]
        result = compute_drift(values, list(values))
        assert result["drift_detected"] is False
        assert result["ks_statistic"] == pytest.approx(0.0)

    def test_same_distribution_different_draws_shows_no_drift(self, rng: random.Random) -> None:
        reference = [rng.gauss(0.5, 0.1) for _ in range(SAMPLE_SIZE)]
        current = [rng.gauss(0.5, 0.1) for _ in range(SAMPLE_SIZE)]
        assert compute_drift(reference, current)["drift_detected"] is False

    def test_shifted_distribution_is_detected(self, rng: random.Random) -> None:
        reference = [rng.gauss(0.3, 0.05) for _ in range(SAMPLE_SIZE)]
        current = [rng.gauss(0.8, 0.05) for _ in range(SAMPLE_SIZE)]
        result = compute_drift(reference, current)
        assert result["drift_detected"] is True
        assert result["p_value"] < 0.05

    def test_larger_shift_raises_ks_statistic(self, rng: random.Random) -> None:
        reference = [rng.gauss(0.5, 0.05) for _ in range(SAMPLE_SIZE)]
        small = [rng.gauss(0.55, 0.05) for _ in range(SAMPLE_SIZE)]
        large = [rng.gauss(0.95, 0.05) for _ in range(SAMPLE_SIZE)]
        assert (
            compute_drift(reference, large)["ks_statistic"]
            > compute_drift(reference, small)["ks_statistic"]
        )

    def test_variance_change_is_detected(self, rng: random.Random) -> None:
        reference = [rng.gauss(0.5, 0.02) for _ in range(SAMPLE_SIZE)]
        current = [rng.gauss(0.5, 0.30) for _ in range(SAMPLE_SIZE)]
        assert compute_drift(reference, current)["drift_detected"] is True

    def test_drift_is_logged_as_a_warning(
        self, rng: random.Random, caplog: pytest.LogCaptureFixture
    ) -> None:
        reference = [rng.gauss(0.2, 0.05) for _ in range(SAMPLE_SIZE)]
        current = [rng.gauss(0.9, 0.05) for _ in range(SAMPLE_SIZE)]
        compute_drift(reference, current)
        assert "Drift detected" in caplog.text

    @pytest.mark.parametrize("size", [0, 1, 4])
    def test_short_reference_returns_insufficient_data(self, size: int) -> None:
        result = compute_drift([0.5] * size, [0.5] * 50)
        assert result["reason"] == "insufficient_data"
        assert result["drift_detected"] is False

    @pytest.mark.parametrize("size", [0, 1, 4])
    def test_short_current_returns_insufficient_data(self, size: int) -> None:
        result = compute_drift([0.5] * 50, [0.5] * size)
        assert result["reason"] == "insufficient_data"

    def test_minimum_sample_size_is_five(self) -> None:
        # Five on each side is enough to run the test rather than bail out.
        assert "reason" not in compute_drift([0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0])

    def test_ks_statistic_is_bounded(self, rng: random.Random) -> None:
        reference = [rng.gauss(0.5, 0.1) for _ in range(SAMPLE_SIZE)]
        current = [rng.gauss(0.9, 0.1) for _ in range(SAMPLE_SIZE)]
        assert 0.0 <= compute_drift(reference, current)["ks_statistic"] <= 1.0

    def test_p_value_is_bounded(self, rng: random.Random) -> None:
        reference = [rng.gauss(0.5, 0.1) for _ in range(SAMPLE_SIZE)]
        current = [rng.gauss(0.6, 0.1) for _ in range(SAMPLE_SIZE)]
        assert 0.0 <= compute_drift(reference, current)["p_value"] <= 1.0

    def test_result_always_carries_the_core_keys(self) -> None:
        for reference, current in (([0.5] * 2, [0.5] * 2), ([0.1] * 50, [0.9] * 50)):
            result = compute_drift(reference, current)
            assert {"ks_statistic", "p_value", "drift_detected"} <= set(result)

    def test_disjoint_ranges_give_maximal_statistic(self) -> None:
        assert compute_drift([0.0] * 50, [1.0] * 50)["ks_statistic"] == pytest.approx(1.0)


class TestDriftSeverity:
    @pytest.mark.parametrize(
        ("statistic", "expected"),
        [
            (0.0, "none"),
            (0.09, "none"),
            (0.1, "mild"),
            (0.24, "mild"),
            (0.25, "moderate"),
            (0.49, "moderate"),
            (0.5, "severe"),
            (1.0, "severe"),
        ],
    )
    def test_severity_boundaries(self, statistic: float, expected: str) -> None:
        assert drift_severity(statistic) == expected

    def test_severity_escalates_monotonically(self) -> None:
        order = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        severities = [order[drift_severity(s / 100)] for s in range(101)]
        assert severities == sorted(severities)

    def test_computed_statistic_maps_to_a_known_severity(self, rng: random.Random) -> None:
        reference = [rng.gauss(0.5, 0.1) for _ in range(SAMPLE_SIZE)]
        current = [rng.gauss(0.7, 0.1) for _ in range(SAMPLE_SIZE)]
        statistic = compute_drift(reference, current)["ks_statistic"]
        assert drift_severity(statistic) in {"none", "mild", "moderate", "severe"}


class TestDriftResultKeys:
    """Verify the dict keys returned by compute_drift."""

    @pytest.mark.parametrize("key", ["drift_detected", "ks_statistic", "p_value"])
    def test_result_has_key(self, rng: random.Random, key: str) -> None:
        """compute_drift result always contains the expected key."""
        values = [rng.gauss(0.5, 0.1) for _ in range(SAMPLE_SIZE)]
        result = compute_drift(values, list(values))
        assert key in result

    def test_ks_statistic_is_float(self, rng: random.Random) -> None:
        values = [rng.gauss(0.5, 0.1) for _ in range(SAMPLE_SIZE)]
        result = compute_drift(values, list(values))
        assert isinstance(result["ks_statistic"], float)

    def test_p_value_between_zero_and_one(self, rng: random.Random) -> None:
        reference = [rng.gauss(0.3, 0.05) for _ in range(SAMPLE_SIZE)]
        current = [rng.gauss(0.7, 0.05) for _ in range(SAMPLE_SIZE)]
        result = compute_drift(reference, current)
        assert 0.0 <= result["p_value"] <= 1.0


@pytest.mark.parametrize(
    "statistic,valid_label",
    [
        (0.0, "none"),
        (0.12, "none"),
        (0.13, "mild"),
        (0.24, "mild"),
        (0.25, "moderate"),
        (0.49, "moderate"),
        (0.5, "severe"),
        (0.99, "severe"),
    ],
)
def test_drift_severity_label_parametrized(statistic: float, valid_label: str) -> None:
    """drift_severity maps each boundary value to the correct label."""
    assert drift_severity(statistic) == valid_label
