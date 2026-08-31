"""Edge-case tests for app/validation.py.

Complements tests/test_validation.py by covering boundary values, type
coercion quirks, and the warning paths that do not affect validity.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.validation import (
    FAULT_TYPES,
    RANGES,
    REQUIRED_FIELDS,
    amplitudes_are_coherent,
    missing_fields,
    normalise_fault_type,
    out_of_range_fields,
    validate_event,
)


def valid_payload(**overrides: Any) -> dict[str, Any]:
    """Build a payload that passes validation, overridable per test."""
    payload: dict[str, Any] = {
        "latitude": 35.0,
        "longitude": 139.0,
        "depth_km": 30.0,
        "station_count": 12,
        "p_wave_amplitude": 100.0,
        "s_wave_amplitude": 180.0,
        "epicentral_distance_km": 45.0,
        "fault_type": "reverse",
    }
    payload.update(overrides)
    return payload


class TestMissingFields:
    def test_complete_payload_has_none(self) -> None:
        assert missing_fields(valid_payload()) == []

    def test_empty_payload_misses_everything(self) -> None:
        assert set(missing_fields({})) == set(REQUIRED_FIELDS)

    def test_reports_only_the_absent_field(self) -> None:
        payload = valid_payload()
        del payload["depth_km"]
        assert missing_fields(payload) == ["depth_km"]

    def test_explicit_none_counts_as_present(self) -> None:
        # Presence is keyed on the field existing, not on its value.
        assert missing_fields(valid_payload(depth_km=None)) == []

    def test_extra_fields_are_ignored(self) -> None:
        assert missing_fields(valid_payload(magnitude_hint=6.1)) == []


class TestOutOfRangeFields:
    def test_valid_payload_has_none(self) -> None:
        assert out_of_range_fields(valid_payload()) == []

    @pytest.mark.parametrize("field", sorted(RANGES))
    def test_lower_bound_is_inclusive(self, field: str) -> None:
        low, _ = RANGES[field]
        assert field not in out_of_range_fields(valid_payload(**{field: low}))

    @pytest.mark.parametrize("field", sorted(RANGES))
    def test_upper_bound_is_inclusive(self, field: str) -> None:
        _, high = RANGES[field]
        assert field not in out_of_range_fields(valid_payload(**{field: high}))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("latitude", 90.1),
            ("latitude", -90.1),
            ("longitude", 180.1),
            ("depth_km", -1.0),
            ("depth_km", 700.1),
            ("station_count", 0),
            ("epicentral_distance_km", -0.1),
        ],
    )
    def test_values_beyond_bounds_are_flagged(self, field: str, value: float) -> None:
        assert field in out_of_range_fields(valid_payload(**{field: value}))

    def test_missing_value_is_not_range_checked(self) -> None:
        payload = valid_payload()
        del payload["latitude"]
        assert "latitude" not in out_of_range_fields(payload)

    def test_none_value_is_not_range_checked(self) -> None:
        assert "latitude" not in out_of_range_fields(valid_payload(latitude=None))

    def test_string_value_is_not_range_checked(self) -> None:
        assert "latitude" not in out_of_range_fields(valid_payload(latitude="35.0"))

    def test_booleans_are_not_treated_as_numbers(self) -> None:
        # bool is a subclass of int; it must not slip through as a valid value.
        assert "station_count" not in out_of_range_fields(valid_payload(station_count=True))

    def test_multiple_violations_all_reported(self) -> None:
        flagged = out_of_range_fields(valid_payload(latitude=200.0, depth_km=-5.0))
        assert set(flagged) == {"latitude", "depth_km"}


class TestNormaliseFaultType:
    @pytest.mark.parametrize("fault_type", FAULT_TYPES)
    def test_canonical_values_pass_through(self, fault_type: str) -> None:
        assert normalise_fault_type(fault_type) == fault_type

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("STRIKE_SLIP", "strike_slip"),
            ("Strike-Slip", "strike_slip"),
            ("strike slip", "strike_slip"),
            ("  reverse  ", "reverse"),
            ("NORMAL", "normal"),
        ],
    )
    def test_case_spacing_and_hyphens_are_normalised(self, raw: str, expected: str) -> None:
        assert normalise_fault_type(raw) == expected

    @pytest.mark.parametrize("raw", ["thrust", "", "  ", "listric", "42"])
    def test_unrecognised_values_fall_back_to_unknown(self, raw: str) -> None:
        assert normalise_fault_type(raw) == "unknown"

    def test_result_is_always_a_known_type(self) -> None:
        for raw in ("reverse", "THRUST", "", "oblique-slip"):
            assert normalise_fault_type(raw) in FAULT_TYPES


class TestAmplitudesAreCoherent:
    def test_typical_pair_is_coherent(self) -> None:
        assert amplitudes_are_coherent(100.0, 180.0) is True

    def test_s_at_exactly_half_p_is_the_boundary(self) -> None:
        assert amplitudes_are_coherent(100.0, 50.0) is True

    def test_s_below_half_p_is_incoherent(self) -> None:
        assert amplitudes_are_coherent(100.0, 49.9) is False

    def test_s_far_above_p_is_coherent(self) -> None:
        assert amplitudes_are_coherent(10.0, 500.0) is True

    @pytest.mark.parametrize(
        ("p_wave", "s_wave"),
        [(0.0, 100.0), (100.0, 0.0), (-1.0, 100.0), (100.0, -1.0), (0.0, 0.0)],
    )
    def test_non_positive_amplitudes_are_incoherent(self, p_wave: float, s_wave: float) -> None:
        assert amplitudes_are_coherent(p_wave, s_wave) is False


class TestValidateEvent:
    def test_good_payload_is_valid_and_quiet(self) -> None:
        result = validate_event(valid_payload())
        assert result["valid"] is True
        assert result["missing"] == []
        assert result["out_of_range"] == []
        assert result["warnings"] == []

    def test_missing_field_invalidates(self) -> None:
        payload = valid_payload()
        del payload["latitude"]
        result = validate_event(payload)
        assert result["valid"] is False
        assert "latitude" in result["missing"]

    def test_out_of_range_field_invalidates(self) -> None:
        result = validate_event(valid_payload(depth_km=5000.0))
        assert result["valid"] is False
        assert "depth_km" in result["out_of_range"]

    def test_transposed_amplitudes_warn_without_invalidating(self) -> None:
        result = validate_event(valid_payload(p_wave_amplitude=200.0, s_wave_amplitude=10.0))
        assert result["valid"] is True
        assert any("amplitude" in w for w in result["warnings"])

    def test_unknown_fault_type_warns_without_invalidating(self) -> None:
        result = validate_event(valid_payload(fault_type="thrust"))
        assert result["valid"] is True
        assert any("fault_type" in w for w in result["warnings"])

    def test_explicit_unknown_fault_type_does_not_warn(self) -> None:
        result = validate_event(valid_payload(fault_type="unknown"))
        assert result["warnings"] == []

    def test_multiple_warnings_accumulate(self) -> None:
        result = validate_event(
            valid_payload(fault_type="thrust", p_wave_amplitude=200.0, s_wave_amplitude=10.0)
        )
        assert len(result["warnings"]) == 2

    def test_empty_payload_reports_all_missing(self) -> None:
        result = validate_event({})
        assert result["valid"] is False
        assert set(result["missing"]) == set(REQUIRED_FIELDS)

    def test_verdict_always_has_the_full_shape(self) -> None:
        for payload in ({}, valid_payload(), valid_payload(latitude=999.0)):
            result = validate_event(payload)
            assert set(result) == {"valid", "missing", "out_of_range", "warnings"}
