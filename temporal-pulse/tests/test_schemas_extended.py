"""Extended tests for temporal-pulse app/schemas.py."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.schemas import AnomalyResult, BatchSensorReadings, SensorReading


class TestSensorReading:
    def test_create_valid(self) -> None:
        r = SensorReading(
            sensor_id="s1",
            timestamp=datetime(2024, 1, 1, 12, 0),
            values={"temp": 72.5, "rpm": 3000.0},
        )
        assert r.sensor_id == "s1"

    def test_rejects_empty_sensor_id(self) -> None:
        with pytest.raises(Exception):
            SensorReading(
                sensor_id="",
                timestamp=datetime.now(),
                values={"temp": 1.0},
            )

    def test_rejects_empty_values(self) -> None:
        with pytest.raises(Exception):
            SensorReading(
                sensor_id="s1",
                timestamp=datetime.now(),
                values={},
            )

    def test_rejects_nan_value(self) -> None:
        import math
        with pytest.raises(Exception):
            SensorReading(
                sensor_id="s1",
                timestamp=datetime.now(),
                values={"temp": math.nan},
            )

    def test_rejects_inf_value(self) -> None:
        import math
        with pytest.raises(Exception):
            SensorReading(
                sensor_id="s1",
                timestamp=datetime.now(),
                values={"temp": math.inf},
            )

    def test_multiple_channels(self) -> None:
        r = SensorReading(
            sensor_id="multi",
            timestamp=datetime.now(),
            values={"a": 1.0, "b": 2.0, "c": 3.0},
        )
        assert len(r.values) == 3


class TestBatchSensorReadings:
    def test_create_valid(self) -> None:
        reading = SensorReading(
            sensor_id="s1", timestamp=datetime.now(), values={"t": 1.0}
        )
        batch = BatchSensorReadings(readings=[reading])
        assert len(batch.readings) == 1

    def test_default_horizon(self) -> None:
        reading = SensorReading(
            sensor_id="s1", timestamp=datetime.now(), values={"t": 1.0}
        )
        batch = BatchSensorReadings(readings=[reading])
        assert batch.horizon == 5

    def test_rejects_empty_readings(self) -> None:
        with pytest.raises(Exception):
            BatchSensorReadings(readings=[])

    def test_rejects_horizon_zero(self) -> None:
        reading = SensorReading(
            sensor_id="s1", timestamp=datetime.now(), values={"t": 1.0}
        )
        with pytest.raises(Exception):
            BatchSensorReadings(readings=[reading], horizon=0)


class TestAnomalyResult:
    def test_create_valid(self) -> None:
        r = AnomalyResult(
            sensor_id="s1",
            timestamp=datetime.now(),
            anomaly_score=0.7,
            is_anomaly=True,
            threshold=0.5,
        )
        assert r.is_anomaly is True

    def test_score_at_boundary(self) -> None:
        r = AnomalyResult(
            sensor_id="s1",
            timestamp=datetime.now(),
            anomaly_score=0.0,
            is_anomaly=False,
            threshold=0.5,
        )
        assert r.anomaly_score == pytest.approx(0.0)
