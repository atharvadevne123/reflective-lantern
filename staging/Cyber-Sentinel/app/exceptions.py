"""Custom exception hierarchy for Cyber-Sentinel."""
from __future__ import annotations


class CyberSentinelError(Exception):
    """Base exception for all Cyber-Sentinel errors."""


class ModelNotTrainedError(CyberSentinelError):
    """Raised when a prediction is requested before the model is trained."""


class FeatureExtractionError(CyberSentinelError):
    """Raised when feature engineering fails on a malformed event."""


class DriftDetectionError(CyberSentinelError):
    """Raised when the KS-test cannot be computed."""


class DatabaseError(CyberSentinelError):
    """Raised on database connectivity or query errors."""
