"""Tests for shared input validation helpers."""

from __future__ import annotations

import pytest

from app.validation import (
    amplitudes_are_coherent,
    missing_fields,
    normalise_fault_type,
    out_of_range_fields,
    validate_event,
)

VALID = {
    "latitude": 35.6,
    "longitude": 139.7,
    "depth_km": 20.0,
    "station_count": 12,
    "p_wave_amplitude": 4.1,
    "s_wave_amplitude": 7.8,
    "epicentral_distance_km": 100.0,
    "fault_type": "reverse",
}


class TestMissingFields:
    def test_complete_payload_has_none(self) -> None:
        assert missing_fields(VALID) == []

    def test_reports_absent_field(self) -> None:
        payload = {k: v for k, v in VALID.items() if k != "depth_km"}
        assert "depth_km" in missing_fields(payload)

    def test_empty_payload_reports_all(self) -> None:
        assert len(missing_fields({})) == 8


class TestOutOfRangeFields:
    def test_valid_payload_is_clean(self) -> None:
        assert out_of_range_fields(VALID) == []

    @pytest.mark.parametrize(
        "field,value",
        [
            ("latitude", 120.0),
            ("longitude", -400.0),
            ("depth_km", 5000.0),
            ("station_count", 0),
            ("epicentral_distance_km", 99999.0),
        ],
    )
    def test_flags_out_of_range(self, field: str, value: float) -> None:
        assert field in out_of_range_fields({**VALID, field: value})

    def test_ignores_non_numeric(self) -> None:
        assert out_of_range_fields({**VALID, "latitude": "north"}) == []

    def test_ignores_booleans(self) -> None:
        assert "station_count" not in out_of_range_fields({**VALID, "station_count": True})


class TestNormaliseFaultType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("REVERSE", "reverse"),
            ("  normal  ", "normal"),
            ("strike-slip", "strike_slip"),
            ("strike slip", "strike_slip"),
            ("nonsense", "unknown"),
        ],
    )
    def test_normalisation(self, raw: str, expected: str) -> None:
        assert normalise_fault_type(raw) == expected


class TestAmplitudesAreCoherent:
    def test_normal_pair_is_coherent(self) -> None:
        assert amplitudes_are_coherent(4.0, 8.0) is True

    def test_transposed_pair_flagged(self) -> None:
        assert amplitudes_are_coherent(10.0, 1.0) is False

    def test_zero_amplitude_rejected(self) -> None:
        assert amplitudes_are_coherent(0.0, 5.0) is False

    def test_negative_amplitude_rejected(self) -> None:
        assert amplitudes_are_coherent(-1.0, 5.0) is False

    def test_boundary_at_half(self) -> None:
        assert amplitudes_are_coherent(10.0, 5.0) is True


class TestValidateEvent:
    def test_valid_payload_passes(self) -> None:
        assert validate_event(VALID)["valid"] is True

    def test_missing_field_fails(self) -> None:
        payload = {k: v for k, v in VALID.items() if k != "latitude"}
        assert validate_event(payload)["valid"] is False

    def test_out_of_range_fails(self) -> None:
        assert validate_event({**VALID, "depth_km": 9000.0})["valid"] is False

    def test_transposed_amplitudes_warn_but_pass(self) -> None:
        result = validate_event({**VALID, "p_wave_amplitude": 20.0, "s_wave_amplitude": 2.0})
        assert result["valid"] is True
        assert any("transposed" in w for w in result["warnings"])

    def test_unknown_fault_type_warns(self) -> None:
        result = validate_event({**VALID, "fault_type": "dragon"})
        assert any("Unrecognised" in w for w in result["warnings"])

    def test_explicit_unknown_does_not_warn(self) -> None:
        result = validate_event({**VALID, "fault_type": "unknown"})
        assert result["warnings"] == []


class TestMissingFieldsParametrized:
    @pytest.mark.parametrize(
        "field",
        ["latitude", "longitude", "depth_km", "station_count", "p_wave_amplitude"],
    )
    def test_each_required_field_reported_when_absent(self, field: str) -> None:
        payload = {k: v for k, v in VALID.items() if k != field}
        assert field in missing_fields(payload)


class TestNormaliseFaultTypeEdgeCases:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("NORMAL", "normal"),
            ("oblique", "oblique"),
            ("unknown", "unknown"),
            ("", "unknown"),
            ("STRIKE_SLIP", "strike_slip"),
        ],
    )
    def test_additional_normalisation_cases(self, raw: str, expected: str) -> None:
        assert normalise_fault_type(raw) == expected


class TestAmplitudesEdgeCases:
    @pytest.mark.parametrize(
        "p_amp,s_amp,expected",
        [
            (1.0, 2.0, True),
            (5.0, 10.0, True),
            (10.0, 4.9, False),
            (0.001, 0.5, True),
        ],
    )
    def test_amplitude_coherence_parametrized(
        self, p_amp: float, s_amp: float, expected: bool
    ) -> None:
        assert amplitudes_are_coherent(p_amp, s_amp) is expected
