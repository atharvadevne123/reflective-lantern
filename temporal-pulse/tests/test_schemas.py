"""Tests for Pydantic schema validation."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pydantic import ValidationError


class TestSensorReadingSchema:
    def test_valid_reading_accepted(self):
        from app.schemas import SensorReading

        r = SensorReading(
            sensor_id="s1",
            timestamp="2026-01-01T00:00:00",
            values={"temp": 20.0},
        )
        assert r.sensor_id == "s1"

    def test_empty_sensor_id_rejected(self):
        from app.schemas import SensorReading

        with pytest.raises(ValidationError):
            SensorReading(sensor_id="", timestamp="2026-01-01T00:00:00", values={"t": 1.0})

    def test_nan_value_rejected(self):
        from app.schemas import SensorReading

        with pytest.raises(ValidationError):
            SensorReading(
                sensor_id="s1",
                timestamp="2026-01-01T00:00:00",
                values={"temp": float("nan")},
            )

    def test_inf_value_rejected(self):
        from app.schemas import SensorReading

        with pytest.raises(ValidationError):
            SensorReading(
                sensor_id="s1",
                timestamp="2026-01-01T00:00:00",
                values={"temp": float("inf")},
            )

    def test_empty_values_rejected(self):
        from app.schemas import SensorReading

        with pytest.raises(ValidationError):
            SensorReading(sensor_id="s1", timestamp="2026-01-01T00:00:00", values={})

    def test_long_sensor_id_rejected(self):
        from app.schemas import SensorReading

        with pytest.raises(ValidationError):
            SensorReading(sensor_id="x" * 65, timestamp="2026-01-01T00:00:00", values={"t": 1.0})


class TestBatchSensorReadings:
    def test_horizon_bounds(self):
        from app.schemas import BatchSensorReadings, SensorReading

        reading = SensorReading(sensor_id="s1", timestamp="2026-01-01T00:00:00", values={"t": 1.0})
        with pytest.raises(ValidationError):
            BatchSensorReadings(readings=[reading], horizon=0)
        with pytest.raises(ValidationError):
            BatchSensorReadings(readings=[reading], horizon=101)

    def test_default_horizon(self):
        from app.schemas import BatchSensorReadings, SensorReading

        reading = SensorReading(sensor_id="s1", timestamp="2026-01-01T00:00:00", values={"t": 1.0})
        batch = BatchSensorReadings(readings=[reading])
        assert batch.horizon == 5


class TestAnomalyResultSchema:
    def test_score_out_of_range_rejected(self):
        from app.schemas import AnomalyResult

        with pytest.raises(ValidationError):
            AnomalyResult(
                sensor_id="s1",
                timestamp="2026-01-01T00:00:00",
                anomaly_score=1.5,
                is_anomaly=True,
                threshold=0.7,
            )

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_valid_scores(self, score):
        from app.schemas import AnomalyResult

        r = AnomalyResult(
            sensor_id="s1",
            timestamp="2026-01-01T00:00:00",
            anomaly_score=score,
            is_anomaly=score >= 0.7,
            threshold=0.7,
        )
        assert r.anomaly_score == score


class TestTrainRequest:
    def test_too_few_readings_rejected(self):
        from app.schemas import SensorReading, TrainRequest

        readings = [
            SensorReading(sensor_id="s1", timestamp="2026-01-01T00:00:00", values={"t": 1.0})
        ] * 10
        with pytest.raises(ValidationError):
            TrainRequest(readings=readings)

    def test_contamination_bounds(self):
        from app.schemas import SensorReading, TrainRequest

        readings = [
            SensorReading(sensor_id="s1", timestamp="2026-01-01T00:00:00", values={"t": 1.0})
        ] * 50
        with pytest.raises(ValidationError):
            TrainRequest(readings=readings, contamination=0.6)


class TestSensorReadingEdgeCases:
    @pytest.mark.parametrize("sensor_id", ["a", "sensor-001", "X" * 50])
    def test_various_sensor_ids_accepted(self, sensor_id: str) -> None:
        from app.schemas import SensorReading

        r = SensorReading(sensor_id=sensor_id, timestamp="2026-01-01T00:00:00", values={"t": 1.0})
        assert r.sensor_id == sensor_id

    def test_empty_values_accepted(self) -> None:
        from app.schemas import SensorReading

        r = SensorReading(sensor_id="s1", timestamp="2026-01-01T00:00:00", values={})
        assert r.values == {}

    def test_multiple_channels_in_values(self) -> None:
        from app.schemas import SensorReading

        vals = {"temperature": 20.0, "vibration": 1.5, "rpm": 1200.0}
        r = SensorReading(sensor_id="s1", timestamp="2026-01-01T00:00:00", values=vals)
        assert r.values == vals
