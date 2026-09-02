"""Feature engineering pipeline for network intrusion detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROTOCOL_TYPES = ["tcp", "udp", "icmp"]
SERVICES = [
    "http", "ftp", "smtp", "ssh", "dns", "ftp_data", "finger", "auth",
    "telnet", "pop_3", "irc", "other",
]
FLAGS = ["SF", "S0", "REJ", "RSTO", "RSTOS0", "SH", "OTH", "S1", "S2", "S3"]


class NetworkFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives lag, rolling-ratio, and interaction features from raw packet fields."""

    def __init__(self) -> None:
        self.protocol_enc = LabelEncoder()
        self.service_enc = LabelEncoder()
        self.flag_enc = LabelEncoder()
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NetworkFeatureEngineer":
        self.protocol_enc.fit(PROTOCOL_TYPES)
        self.service_enc.fit(SERVICES)
        self.flag_enc.fit(FLAGS)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()

        # Categorical encodings
        df["protocol_enc"] = df["protocol_type"].apply(
            lambda v: self.protocol_enc.transform([v])[0] if v in self.protocol_enc.classes_ else 0
        )
        df["service_enc"] = df["service"].apply(
            lambda v: self.service_enc.transform([v])[0] if v in self.service_enc.classes_ else 0
        )
        df["flag_enc"] = df["flag"].apply(
            lambda v: self.flag_enc.transform([v])[0] if v in self.flag_enc.classes_ else 0
        )

        # Ratio features
        df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
        df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]

        # Log-transforms for heavy-tailed distributions
        df["log_src_bytes"] = np.log1p(df["src_bytes"])
        df["log_dst_bytes"] = np.log1p(df["dst_bytes"])
        df["log_duration"] = np.log1p(df["duration"])

        # Interaction features
        df["bytes_per_second"] = df["total_bytes"] / (df["duration"] + 1)
        df["src_dst_diff"] = df["src_bytes"] - df["dst_bytes"]

        # Rolling stats (within a sorted window proxy for connection bursts)
        df["rolling_src_mean"] = df["src_bytes"].rolling(5, min_periods=1).mean()
        df["rolling_src_std"] = df["src_bytes"].rolling(5, min_periods=1).std().fillna(0)

        feature_cols = [
            "protocol_enc", "service_enc", "flag_enc",
            "src_bytes", "dst_bytes", "duration",
            "byte_ratio", "total_bytes",
            "log_src_bytes", "log_dst_bytes", "log_duration",
            "bytes_per_second", "src_dst_diff",
            "rolling_src_mean", "rolling_src_std",
        ]
        return df[feature_cols].values.astype(np.float64)


def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        ("engineer", NetworkFeatureEngineer()),
        ("scaler", StandardScaler()),
    ])


FEATURE_NAMES = [
    "protocol_enc", "service_enc", "flag_enc",
    "src_bytes", "dst_bytes", "duration",
    "byte_ratio", "total_bytes",
    "log_src_bytes", "log_dst_bytes", "log_duration",
    "bytes_per_second", "src_dst_diff",
    "rolling_src_mean", "rolling_src_std",
]


def make_sample_df(
    src_bytes: float = 100.0,
    dst_bytes: float = 200.0,
    duration: float = 1.0,
    protocol_type: str = "tcp",
    service: str = "http",
    flag: str = "SF",
) -> pd.DataFrame:
    return pd.DataFrame([{
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "duration": duration,
        "protocol_type": protocol_type,
        "service": service,
        "flag": flag,
    }])
