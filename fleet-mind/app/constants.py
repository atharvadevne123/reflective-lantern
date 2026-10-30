"""Domain constants for Fleet-Mind fleet telemetry analysis."""

from __future__ import annotations

# Brake pad wear constants (mm)
BRAKE_PAD_NEW_MM: float = 12.0
BRAKE_PAD_WORN_THRESHOLD_MM: float = 3.5
BRAKE_PAD_CRITICAL_MM: float = 2.0

# Engine temperature thresholds (°C)
ENGINE_TEMP_NORMAL_MAX_C: float = 95.0
ENGINE_TEMP_WARNING_C: float = 105.0
ENGINE_TEMP_CRITICAL_C: float = 115.0

# Oil pressure thresholds (bar)
OIL_PRESSURE_MIN_BAR: float = 1.5
OIL_PRESSURE_CRITICAL_BAR: float = 0.8

# Battery voltage thresholds (V)
BATTERY_VOLTAGE_NOMINAL_V: float = 12.6
BATTERY_VOLTAGE_LOW_V: float = 12.0
BATTERY_VOLTAGE_CRITICAL_V: float = 10.5

# Tyre pressure nominal and tolerance (bar)
TIRE_PRESSURE_NOMINAL_BAR: float = 2.4
TIRE_PRESSURE_LOW_BAR: float = 1.8
TIRE_PRESSURE_HIGH_BAR: float = 3.0

# Remaining useful life estimation (km per mm of brake pad)
RUL_KM_PER_MM_BRAKE: float = 5_000.0

# Drift monitoring
DRIFT_BASELINE_SAMPLES: int = 2000
DRIFT_MIN_OBSERVATIONS: int = 50

# Rate limiting defaults
DEFAULT_RATE_LIMIT_RPM: int = 100

# Feature scaler warm-up reads
SCALER_WARMUP_ROWS: int = 10
