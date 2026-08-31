"""KS-test drift detection and prediction logging for Ops-Vision."""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TypedDict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class MetricSample(TypedDict, total=False):
    """Type alias for a raw SRE metrics observation dict."""

    cpu_usage_pct: float
    memory_usage_pct: float
    error_rate_per_min: float
    latency_p99_ms: float
    request_rate_per_sec: float
    disk_io_util_pct: float

DRIFT_THRESHOLD: float = 0.05
REFERENCE_WINDOW_SIZE: int = 1000
CURRENT_WINDOW_SIZE: int = 200

FEATURE_COLS: list[str] = [
    "cpu_usage_pct",
    "memory_usage_pct",
    "error_rate_per_min",
    "latency_p99_ms",
    "request_rate_per_sec",
    "disk_io_util_pct",
]


@dataclass
class DriftResult:
    """Holds KS-test drift detection output for a single feature.

    Attributes:
        feature_name: Name of the feature tested.
        ks_statistic: Kolmogorov-Smirnov test statistic.
        p_value: p-value from the KS test.
        drifted: True if p_value < DRIFT_THRESHOLD.
        timestamp: UTC datetime of the test.
    """

    feature_name: str
    ks_statistic: float
    p_value: float
    drifted: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DriftMonitor:
    """Monitors feature distributions for covariate shift using the KS test.

    Maintains a sliding reference window and a current production window.
    When the current window fills up, it runs KS tests against the reference
    and emits DriftResult objects for each feature.
    """

    def __init__(
        self,
        reference_window_size: int = REFERENCE_WINDOW_SIZE,
        current_window_size: int = CURRENT_WINDOW_SIZE,
        threshold: float = DRIFT_THRESHOLD,
    ) -> None:
        """Initialise the monitor with empty windows.

        Args:
            reference_window_size: Max samples in the reference window.
            current_window_size: Samples to accumulate before drift check.
            threshold: p-value threshold below which drift is flagged.
        """
        self.reference_window_size = reference_window_size
        self.current_window_size = current_window_size
        self.threshold = threshold
        self._reference: deque[dict] = deque(maxlen=reference_window_size)
        self._current: deque[dict] = deque(maxlen=current_window_size)

    def update_reference(self, samples: list[dict]) -> None:
        """Add samples to the reference distribution window.

        Args:
            samples: List of metric dicts (one per observation).
        """
        for s in samples:
            self._reference.append(s)
        logger.info(
            "Reference window updated: %d samples", len(self._reference)
        )

    def record(self, sample: dict) -> Optional[list[DriftResult]]:
        """Record a production sample and trigger drift check when window fills.

        Args:
            sample: Metric dict for a single observation.

        Returns:
            List of DriftResult objects if a check was triggered, else None.
        """
        self._current.append(sample)
        if len(self._current) >= self.current_window_size:
            results = self.check_drift()
            self._current.clear()
            return results
        return None

    def check_drift(self) -> list[DriftResult]:
        """Run KS tests for all features and return drift results.

        Returns:
            List of DriftResult, one per feature.

        Raises:
            ValueError: If the reference window is empty.
        """
        if not self._reference:
            raise ValueError("Reference window is empty — cannot test for drift")

        ref_arrays: dict[str, np.ndarray] = {
            col: np.array([s.get(col, 0.0) for s in self._reference])
            for col in FEATURE_COLS
        }
        cur_arrays: dict[str, np.ndarray] = {
            col: np.array([s.get(col, 0.0) for s in self._current])
            for col in FEATURE_COLS
        }

        results: list[DriftResult] = []
        for col in FEATURE_COLS:
            ks_stat, p_val = stats.ks_2samp(ref_arrays[col], cur_arrays[col])
            drifted = bool(p_val < self.threshold)
            result = DriftResult(
                feature_name=col,
                ks_statistic=float(ks_stat),
                p_value=float(p_val),
                drifted=drifted,
            )
            results.append(result)
            if drifted:
                logger.warning(
                    "DRIFT DETECTED on %s: KS=%.4f p=%.6f", col, ks_stat, p_val
                )
            else:
                logger.debug(
                    "No drift on %s: KS=%.4f p=%.6f", col, ks_stat, p_val
                )

        return results

    @property
    def reference_size(self) -> int:
        """Number of samples in the reference window."""
        return len(self._reference)

    @property
    def current_size(self) -> int:
        """Number of samples accumulated in the current window."""
        return len(self._current)


_monitor_singleton: Optional[DriftMonitor] = None


def get_monitor() -> DriftMonitor:
    """Return the global DriftMonitor singleton (lazy init)."""
    global _monitor_singleton
    if _monitor_singleton is None:
        _monitor_singleton = DriftMonitor()
        logger.info("DriftMonitor singleton initialised")
    return _monitor_singleton
