"""Extended tests for Pydantic validation models including new schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.validation import (
    OHLCVRow,
    TrainRequest,
    TrainResponse,
    VersionResponse,
)


class TestOHLCVRow:
    def test_valid_row(self) -> None:
        row = OHLCVRow(open=100, high=105, low=98, close=102, volume=10000)
        assert row.close == 102

    def test_high_less_than_low_raises(self) -> None:
        with pytest.raises(ValidationError):
            OHLCVRow(open=100, high=90, low=95, close=91, volume=100)

    def test_negative_open_raises(self) -> None:
        with pytest.raises(ValidationError):
            OHLCVRow(open=-10, high=5, low=3, close=4, volume=100)

    def test_zero_volume_valid(self) -> None:
        row = OHLCVRow(open=100, high=105, low=95, close=100, volume=0)
        assert row.volume == 0

    @pytest.mark.parametrize("field", ["open", "high", "low", "close"])
    def test_zero_price_raises(self, field: str) -> None:
        data = {"open": 100, "high": 105, "low": 95, "close": 100, "volume": 100}
        data[field] = 0
        with pytest.raises(ValidationError):
            OHLCVRow(**data)


class TestTrainRequest:
    def test_default_values(self) -> None:
        req = TrainRequest()
        assert req.n_samples == 1000
        assert req.force is False

    def test_custom_n_samples(self) -> None:
        req = TrainRequest(n_samples=5000)
        assert req.n_samples == 5000

    def test_force_true(self) -> None:
        req = TrainRequest(force=True)
        assert req.force is True

    def test_n_samples_below_min_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrainRequest(n_samples=50)

    def test_n_samples_above_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrainRequest(n_samples=100_000)

    @pytest.mark.parametrize("n", [100, 500, 1000, 50_000])
    def test_boundary_values_accepted(self, n: int) -> None:
        req = TrainRequest(n_samples=n)
        assert req.n_samples == n


class TestTrainResponse:
    def test_build_response(self) -> None:
        resp = TrainResponse(
            run_id="abc123",
            auc_mean=0.75,
            auc_std=0.02,
            n_samples=1000,
            n_features=22,
            promoted=True,
        )
        assert resp.promoted is True

    def test_promoted_default_false(self) -> None:
        resp = TrainResponse(
            run_id="run1",
            auc_mean=0.8,
            auc_std=0.01,
            n_samples=500,
            n_features=10,
        )
        assert resp.promoted is False


class TestVersionResponse:
    def test_build_version_response(self) -> None:
        resp = VersionResponse(version="1.0.0", python_version="3.11", model_loaded=True)
        assert resp.version == "1.0.0"
        assert resp.model_loaded is True

    def test_model_loaded_false(self) -> None:
        resp = VersionResponse(version="1.0.0", python_version="3.11", model_loaded=False)
        assert resp.model_loaded is False
