"""Feature engineering pipeline for network intrusion detection."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROTOCOL_MAP: dict[str, int] = {"tcp": 0, "udp": 1, "icmp": 2}
FLAG_MAP: dict[str, int] = {
    "SF": 0,
    "S0": 1,
    "REJ": 2,
    "RSTO": 3,
    "RSTR": 4,
    "SH": 5,
    "S1": 6,
    "S2": 7,
    "S3": 8,
    "OTH": 9,
}
HIGH_RISK_SERVICES: set[str] = {
    "telnet",
    "ftp",
    "smtp",
    "finger",
    "auth",
    "shell",
    "exec",
    "login",
    "pop_3",
    "imap4",
    "netbios_ssn",
}

FEATURE_NAMES: list[str] = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "count",
    "srv_count",
    "serror_rate",
    "rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "protocol_encoded",
    "flag_encoded",
    "service_risk_score",
    "bytes_ratio",
    "total_bytes",
    "bytes_per_second",
    "error_rate_combined",
    "connection_density",
]


class NetworkFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transforms raw network flow dicts into a numeric feature matrix."""

    def fit(self, X: Any, y: Any = None) -> "NetworkFeatureEngineer":
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Convert list of flow dicts to feature matrix."""
        if isinstance(X, pd.DataFrame):
            records = X.to_dict("records")
        elif isinstance(X, list):
            records = X
        else:
            records = [X]

        rows = [self._engineer_one(r) for r in records]
        if not rows:
            # Keep the matrix 2D so downstream estimators get a valid shape
            # instead of a confusing "Expected 2D array" error.
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def _engineer_one(self, flow: dict[str, Any]) -> list[float]:
        duration = float(flow.get("duration", 0.0))
        src_bytes = float(flow.get("src_bytes", 0.0))
        dst_bytes = float(flow.get("dst_bytes", 0.0))
        land = float(flow.get("land", 0))
        wrong_fragment = float(flow.get("wrong_fragment", 0))
        urgent = float(flow.get("urgent", 0))
        hot = float(flow.get("hot", 0))
        num_failed = float(flow.get("num_failed_logins", 0))
        logged_in = float(flow.get("logged_in", 0))
        num_comp = float(flow.get("num_compromised", 0))
        count = float(flow.get("count", 1))
        srv_count = float(flow.get("srv_count", 1))
        serror_rate = float(flow.get("serror_rate", 0.0))
        rerror_rate = float(flow.get("rerror_rate", 0.0))
        same_srv_rate = float(flow.get("same_srv_rate", 1.0))
        diff_srv_rate = float(flow.get("diff_srv_rate", 0.0))
        dst_host_count = float(flow.get("dst_host_count", 1))
        dst_host_srv = float(flow.get("dst_host_srv_count", 1))
        dst_host_same_srv = float(flow.get("dst_host_same_srv_rate", 1.0))
        dst_host_diff_srv = float(flow.get("dst_host_diff_srv_rate", 0.0))

        protocol = str(flow.get("protocol_type", "tcp")).lower()
        protocol_encoded = float(PROTOCOL_MAP.get(protocol, 0))

        flag = str(flow.get("flag", "SF"))
        flag_encoded = float(FLAG_MAP.get(flag, 0))

        service = str(flow.get("service", "http")).lower()
        service_risk = 1.0 if service in HIGH_RISK_SERVICES else 0.0

        # Derived features
        total_bytes = src_bytes + dst_bytes
        bytes_ratio = src_bytes / (dst_bytes + 1e-9)
        bytes_per_second = total_bytes / (duration + 1e-9)
        error_rate_combined = (serror_rate + rerror_rate) / 2.0
        connection_density = count / (dst_host_count + 1e-9)

        return [
            duration,
            src_bytes,
            dst_bytes,
            land,
            wrong_fragment,
            urgent,
            hot,
            num_failed,
            logged_in,
            num_comp,
            count,
            srv_count,
            serror_rate,
            rerror_rate,
            same_srv_rate,
            diff_srv_rate,
            dst_host_count,
            dst_host_srv,
            dst_host_same_srv,
            dst_host_diff_srv,
            protocol_encoded,
            flag_encoded,
            service_risk,
            bytes_ratio,
            total_bytes,
            bytes_per_second,
            error_rate_combined,
            connection_density,
        ]


def build_feature_pipeline() -> Pipeline:
    """Return an sklearn pipeline that engineers + scales features."""
    return Pipeline(
        [
            ("engineer", NetworkFeatureEngineer()),
            ("scaler", StandardScaler()),
        ]
    )


def generate_synthetic_dataset(
    n_samples: int = 2000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic network flow data for training and testing.

    Args:
        n_samples: Total rows to generate. Split evenly across the five
            classes, so the result is rounded down to a multiple of five
            with a floor of one sample per class.
        seed: Seed for the random number generator.

    Returns:
        X (feature matrix) and y (0=normal, 1=dos, 2=probe, 3=r2l, 4=u2r).
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    labels: list[int] = []

    # Always emit at least one row per class — an empty dataset would blow up
    # downstream with an opaque shape error rather than a useful message.
    per_class = max(1, n_samples // 5)

    # Normal traffic
    for _ in range(per_class):
        records.append(
            {
                "duration": rng.exponential(10),
                "src_bytes": rng.integers(200, 5000),
                "dst_bytes": rng.integers(200, 8000),
                "protocol_type": rng.choice(["tcp", "udp"]),
                "flag": rng.choice(["SF", "S1"]),
                "service": rng.choice(["http", "ftp_data", "smtp"]),
                "logged_in": 1,
                "count": rng.integers(1, 10),
                "serror_rate": rng.uniform(0, 0.1),
                "rerror_rate": rng.uniform(0, 0.1),
                "same_srv_rate": rng.uniform(0.8, 1.0),
                "dst_host_count": rng.integers(1, 20),
            }
        )
        labels.append(0)

    # DoS: high connection count, large src_bytes, high serror_rate
    for _ in range(per_class):
        records.append(
            {
                "duration": rng.exponential(0.5),
                "src_bytes": rng.integers(0, 100),
                "dst_bytes": 0,
                "protocol_type": "tcp",
                "flag": rng.choice(["S0", "REJ"]),
                "service": "http",
                "logged_in": 0,
                "count": rng.integers(200, 512),
                "serror_rate": rng.uniform(0.8, 1.0),
                "rerror_rate": rng.uniform(0.0, 0.2),
                "same_srv_rate": rng.uniform(0.9, 1.0),
                "dst_host_count": rng.integers(200, 256),
            }
        )
        labels.append(1)

    # Probe: port scanning
    for _ in range(per_class):
        records.append(
            {
                "duration": rng.exponential(1),
                "src_bytes": rng.integers(0, 500),
                "dst_bytes": rng.integers(0, 200),
                "protocol_type": rng.choice(["tcp", "icmp"]),
                "flag": rng.choice(["S0", "RSTO", "OTH"]),
                "service": rng.choice(["private", "domain_u", "auth"]),
                "logged_in": 0,
                "count": rng.integers(10, 100),
                "serror_rate": rng.uniform(0.5, 1.0),
                "rerror_rate": rng.uniform(0.0, 0.5),
                "diff_srv_rate": rng.uniform(0.5, 1.0),
                "dst_host_count": rng.integers(50, 256),
            }
        )
        labels.append(2)

    # R2L: remote to local
    for _ in range(per_class):
        records.append(
            {
                "duration": rng.exponential(50),
                "src_bytes": rng.integers(500, 20000),
                "dst_bytes": rng.integers(100, 5000),
                "protocol_type": "tcp",
                "flag": "SF",
                "service": rng.choice(["telnet", "ftp", "smtp"]),
                "logged_in": rng.integers(0, 2),
                "num_failed_logins": rng.integers(1, 10),
                "count": rng.integers(1, 5),
                "serror_rate": rng.uniform(0, 0.3),
                "dst_host_count": rng.integers(1, 20),
            }
        )
        labels.append(3)

    # U2R: user to root
    for _ in range(per_class):
        records.append(
            {
                "duration": rng.exponential(100),
                "src_bytes": rng.integers(1000, 50000),
                "dst_bytes": rng.integers(500, 10000),
                "protocol_type": "tcp",
                "flag": "SF",
                "service": rng.choice(["shell", "exec", "login"]),
                "logged_in": 1,
                "hot": rng.integers(10, 100),
                "num_compromised": rng.integers(1, 20),
                "count": rng.integers(1, 3),
                "serror_rate": rng.uniform(0, 0.2),
                "dst_host_count": rng.integers(1, 10),
            }
        )
        labels.append(4)

    eng = NetworkFeatureEngineer()
    X = eng.transform(records)
    return X, np.array(labels)
