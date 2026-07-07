"""Feature engineering pipeline for network traffic data."""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROTOCOL_MAP: dict[str, int] = {"tcp": 0, "udp": 1, "icmp": 2, "other": 3}
ATTACK_LABELS: list[str] = ["normal", "dos", "probe", "r2l", "u2r"]

FEATURE_NAMES: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "count",
    "srv_count",
    "serror_rate",
    "rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
]


def encode_protocol(protocol: str) -> int:
    """Map protocol string to integer code."""
    return PROTOCOL_MAP.get(protocol.lower(), PROTOCOL_MAP["other"])


def compute_byte_ratio(src_bytes: float, dst_bytes: float) -> float:
    """Return log-ratio of src to dst bytes, avoiding division by zero."""
    dst = max(dst_bytes, 1.0)
    return math.log1p(src_bytes) - math.log1p(dst)


def compute_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy of a count distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def build_feature_vector(raw: dict[str, Any]) -> list[float]:
    """Convert a raw network event dict into a numeric feature vector."""
    duration = float(raw.get("duration_ms", 0.0))
    protocol = encode_protocol(str(raw.get("protocol", "other")))
    src_bytes = float(raw.get("src_bytes", raw.get("payload_bytes", 0)))
    dst_bytes = float(raw.get("dst_bytes", 0))
    byte_ratio = compute_byte_ratio(src_bytes, dst_bytes)
    flags = str(raw.get("flags", "")).upper()
    syn = 1 if "S" in flags else 0
    fin = 1 if "F" in flags else 0
    rst = 1 if "R" in flags else 0
    ack = 1 if "A" in flags else 0

    return [
        math.log1p(duration),
        float(protocol),
        math.log1p(src_bytes),
        math.log1p(dst_bytes),
        byte_ratio,
        float(syn),
        float(fin),
        float(rst),
        float(ack),
        float(raw.get("wrong_fragment", 0)),
        float(raw.get("urgent", 0)),
        float(raw.get("hot", 0)),
        float(raw.get("num_failed_logins", 0)),
        float(raw.get("logged_in", 0)),
        float(raw.get("num_compromised", 0)),
        float(raw.get("root_shell", 0)),
        float(raw.get("su_attempted", 0)),
        float(raw.get("num_root", 0)),
        float(raw.get("num_file_creations", 0)),
        float(raw.get("count", 0)),
        float(raw.get("srv_count", 0)),
        float(raw.get("serror_rate", 0.0)),
        float(raw.get("rerror_rate", 0.0)),
        float(raw.get("same_srv_rate", 1.0)),
        float(raw.get("diff_srv_rate", 0.0)),
    ]


def generate_synthetic_data(n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic network traffic data for training."""
    rng = np.random.default_rng(42)
    n_features = 25
    n_classes = 5

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    samples_per_class = n_samples // n_classes
    for cls in range(n_classes):
        mean = rng.uniform(-1, 1, n_features) * (cls + 1)
        cov = np.eye(n_features) * rng.uniform(0.5, 1.5)
        X_cls = rng.multivariate_normal(mean, cov, samples_per_class)
        y_cls = np.full(samples_per_class, cls)
        X_parts.append(X_cls)
        y_parts.append(y_cls)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def build_preprocessing_pipeline() -> Pipeline:
    """Return a sklearn Pipeline with scaler for network features."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )


def validate_feature_vector(features: list[float]) -> list[float]:
    """Validate and clip feature values to finite range.

    Args:
        features: Raw feature vector of length 25.

    Returns:
        Cleaned feature vector with NaN/Inf replaced by 0.

    Raises:
        ValueError: If features is empty or has wrong length.
    """
    if not features:
        raise ValueError("Feature vector must not be empty")
    if len(features) != 25:
        raise ValueError(f"Expected 25 features, got {len(features)}")
    return [0.0 if not math.isfinite(v) else v for v in features]


def features_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix from a DataFrame of network events."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].fillna(0.0).values
