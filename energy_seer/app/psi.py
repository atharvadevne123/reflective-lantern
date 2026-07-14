"""Population Stability Index (PSI) for feature distribution drift detection."""
from __future__ import annotations

import numpy as np


def compute_psi(
    reference: list[float],
    current: list[float],
    bins: int = 10,
    eps: float = 1e-6,
) -> dict:
    """Compute PSI between reference and current distributions.

    PSI < 0.1  → no significant change
    PSI < 0.25 → moderate change
    PSI >= 0.25 → significant change
    """
    if len(reference) < 5 or len(current) < 5:
        return {"psi": 0.0, "drift_level": "insufficient_data"}

    lo = min(min(reference), min(current))
    hi = max(max(reference), max(current))
    bin_edges = np.linspace(lo, hi, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts + eps) / (len(reference) + eps * bins)
    cur_pct = (cur_counts + eps) / (len(current) + eps * bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    level = (
        "stable" if psi < 0.1
        else "moderate" if psi < 0.25
        else "significant"
    )
    return {"psi": round(psi, 6), "drift_level": level}
