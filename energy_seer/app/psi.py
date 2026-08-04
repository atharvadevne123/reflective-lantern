"""Population Stability Index (PSI) for feature distribution drift detection."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

PSI_STABLE_THRESHOLD: float = 0.1
PSI_MODERATE_THRESHOLD: float = 0.25


def _drift_level(psi: float) -> str:
    """Map a PSI value to a human-readable drift level."""
    if psi < PSI_STABLE_THRESHOLD:
        return "stable"
    if psi < PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "significant"


def compute_psi(
    reference: list[float],
    current: list[float],
    bins: int = 10,
    eps: float = 1e-6,
) -> dict[str, object]:
    """Compute Population Stability Index between reference and current distributions.

    PSI < 0.1   → stable (no significant change)
    PSI < 0.25  → moderate change
    PSI >= 0.25 → significant drift — model retraining recommended

    Args:
        reference: Historical baseline sample (requires ≥ 5 observations).
        current: Recent production sample (requires ≥ 5 observations).
        bins: Number of histogram bins for discretisation (default 10).
        eps: Small value added to bin counts to avoid log(0).

    Returns:
        Dict with 'psi' (float), 'drift_level' (str), and 'bins_used' (int).
        Returns ``{"psi": 0.0, "drift_level": "insufficient_data"}`` when
        either sample has fewer than 5 observations.
    """
    if len(reference) < 5 or len(current) < 5:
        logger.debug(
            "compute_psi: insufficient data (ref=%d, cur=%d)", len(reference), len(current)
        )
        return {"psi": 0.0, "drift_level": "insufficient_data", "bins_used": 0}

    lo = min(min(reference), min(current))
    hi = max(max(reference), max(current))

    if abs(hi - lo) < 1e-12:
        return {"psi": 0.0, "drift_level": "stable", "bins_used": bins}

    bin_edges = np.linspace(lo, hi, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts + eps) / (len(reference) + eps * bins)
    cur_pct = (cur_counts + eps) / (len(current) + eps * bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    level = _drift_level(psi)

    if level != "stable":
        logger.warning("PSI drift detected: psi=%.4f level=%s", psi, level)

    return {"psi": round(psi, 6), "drift_level": level, "bins_used": bins}


def psi_report(
    feature_distributions: dict[str, tuple[list[float], list[float]]],
    bins: int = 10,
) -> dict[str, dict[str, object]]:
    """Compute PSI for multiple features and return a combined drift report.

    Args:
        feature_distributions: Mapping of feature name to (reference, current)
            sample tuples.
        bins: Number of histogram bins passed to :func:`compute_psi`.

    Returns:
        Dict mapping feature name to its PSI result dict.  Also includes a
        'summary' key with overall 'max_psi', 'drifted_features' count, and
        'worst_feature'.
    """
    results: dict[str, dict[str, object]] = {}
    for feature, (ref, cur) in feature_distributions.items():
        results[feature] = compute_psi(ref, cur, bins=bins)

    if results:
        feature_psi = {
            k: float(v["psi"]) for k, v in results.items() if isinstance(v.get("psi"), float)
        }  # type: ignore[arg-type]
        worst = max(feature_psi, key=lambda k: feature_psi[k]) if feature_psi else None
        drifted = sum(
            1
            for v in results.values()
            if v.get("drift_level") not in ("stable", "insufficient_data")
        )
        results["summary"] = {
            "max_psi": round(max(feature_psi.values(), default=0.0), 6),
            "drifted_features": drifted,
            "worst_feature": worst,
        }

    return results


__all__ = [
    "PSI_MODERATE_THRESHOLD",
    "PSI_STABLE_THRESHOLD",
    "compute_psi",
    "psi_report",
]
