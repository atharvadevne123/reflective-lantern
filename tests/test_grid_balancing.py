"""Tests for app.grid_balancing module."""

from __future__ import annotations

import pytest

from app.grid_balancing import (
    AncillaryServiceBid,
    FrequencyEvent,
    NOMINAL_FREQUENCY_HZ,
    ancillary_service_value,
    classify_frequency_event,
    droop_response_kw,
    frequency_deviation,
    inertia_constant,
    rocof,
    swing_equation_rocof,
)


class TestFrequencyDeviation:
    def test_positive_deviation(self) -> None:
        assert frequency_deviation(50.2) == pytest.approx(0.2)

    def test_negative_deviation(self) -> None:
        assert frequency_deviation(49.8) == pytest.approx(-0.2)

    def test_nominal_returns_zero(self) -> None:
        assert frequency_deviation(50.0) == pytest.approx(0.0)

    def test_zero_nominal_raises(self) -> None:
        with pytest.raises(ValueError):
            frequency_deviation(50.0, nominal_hz=0.0)

    @pytest.mark.parametrize("measured,expected", [(50.5, 0.5), (49.5, -0.5)])
    def test_parametrized(self, measured: float, expected: float) -> None:
        assert frequency_deviation(measured) == pytest.approx(expected)


class TestRoCOF:
    def test_declining_frequency(self) -> None:
        samples = [50.0, 49.9, 49.8, 49.7]
        rate = rocof(samples, 0.1)
        assert rate < 0

    def test_stable_frequency(self) -> None:
        samples = [50.0, 50.0, 50.0, 50.0]
        rate = rocof(samples, 0.1)
        assert rate == pytest.approx(0.0, abs=1e-6)

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            rocof([50.0], 0.1)

    def test_zero_interval_raises(self) -> None:
        with pytest.raises(ValueError):
            rocof([50.0, 49.9], 0.0)

    def test_rising_frequency_positive_rocof(self) -> None:
        samples = [50.0, 50.1, 50.2, 50.3]
        rate = rocof(samples, 0.1)
        assert rate > 0


class TestDroopResponse:
    def test_under_frequency_injects_power(self) -> None:
        response = droop_response_kw(-0.5, 100.0)
        assert response > 0

    def test_over_frequency_absorbs_power(self) -> None:
        response = droop_response_kw(0.5, 100.0)
        assert response < 0

    def test_zero_deviation_zero_response(self) -> None:
        assert droop_response_kw(0.0, 100.0) == pytest.approx(0.0)

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            droop_response_kw(-0.5, 0.0)

    def test_zero_droop_raises(self) -> None:
        with pytest.raises(ValueError):
            droop_response_kw(-0.5, 100.0, droop_percent=0.0)

    def test_clipped_at_rated_capacity(self) -> None:
        response = droop_response_kw(-10.0, 100.0)
        assert abs(response) <= 100.0


class TestInertiaConstant:
    def test_basic(self) -> None:
        h = inertia_constant(3000.0, 1000.0)
        assert h == pytest.approx(3.0)

    def test_zero_energy_raises(self) -> None:
        with pytest.raises(ValueError):
            inertia_constant(0.0, 1000.0)

    def test_zero_rating_raises(self) -> None:
        with pytest.raises(ValueError):
            inertia_constant(3000.0, 0.0)


class TestSwingEquationRoCOF:
    def test_power_deficit_negative_rocof(self) -> None:
        rate = swing_equation_rocof(-100.0, 10000.0)
        assert rate < 0

    def test_power_surplus_positive_rocof(self) -> None:
        rate = swing_equation_rocof(100.0, 10000.0)
        assert rate > 0

    def test_balanced_system_zero_rocof(self) -> None:
        assert swing_equation_rocof(0.0, 10000.0) == pytest.approx(0.0)

    def test_zero_inertia_raises(self) -> None:
        with pytest.raises(ValueError):
            swing_equation_rocof(-100.0, 0.0)


class TestClassifyFrequencyEvent:
    def test_normal_event(self) -> None:
        event = classify_frequency_event(50.0, 0.0)
        assert event.severity == "normal"

    def test_severe_event(self) -> None:
        event = classify_frequency_event(48.0, 3.0)
        assert event.severity == "severe"

    def test_returns_dataclass(self) -> None:
        event = classify_frequency_event(49.8, -0.5)
        assert isinstance(event, FrequencyEvent)

    def test_primary_response_required_below_deadband(self) -> None:
        event = classify_frequency_event(49.95, -0.1)
        assert not event.primary_response_required

    def test_primary_response_required_above_deadband(self) -> None:
        event = classify_frequency_event(49.8, -0.5)
        assert event.primary_response_required


class TestAncillaryServiceValue:
    def test_returns_dataclass(self) -> None:
        bid = ancillary_service_value("bat1", 10.0, "FFR", 0.5, 30.0, 50.0)
        assert isinstance(bid, AncillaryServiceBid)

    def test_total_value(self) -> None:
        bid = ancillary_service_value("bat1", 10.0, "PFR", 1.0, 15.0, 30.0)
        assert bid.total_value == pytest.approx(300.0)

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            ancillary_service_value("bat1", 0.0, "FFR", 0.5, 30.0, 50.0)

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValueError):
            ancillary_service_value("bat1", 10.0, "FFR", 0.5, 30.0, -10.0)
