"""Regression tests guarding against train/serve feature skew.

The model is fitted on batches but served one connection at a time. Any
feature whose value depends on the number of rows in the frame will take a
different value in training than in serving, which silently pushes every
served request off the training manifold. Rolling-window statistics are the
classic instance: on a one-row frame ``rolling(5).mean()`` collapses to the
row's own value and ``rolling(5).std()`` is NaN, filled with 0.

These tests pin the invariant that scoring a row alone and scoring it inside
a batch agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.anomaly import batch_anomaly_rate, score_anomaly, train_anomaly_detector
from app.features import FEATURE_NAMES, NetworkFeatureEngineer
from app.model import generate_synthetic_data

ROLLING_FEATURES = ["rolling_src_mean", "rolling_src_std"]


@pytest.fixture(scope="module")
def fitted_engineer():
    X, _ = generate_synthetic_data(500)
    eng = NetworkFeatureEngineer()
    eng.fit(X)
    return eng, X


def test_rolling_std_is_never_degenerate_on_single_row(fitted_engineer):
    """A one-row frame must not emit rolling_src_std == 0.

    Zero is the value the naive ``.fillna(0)`` produced for *every* served
    request, while the training mean was ~427 — the exact skew this guards.
    """
    eng, _ = fitted_engineer
    _, X = fitted_engineer
    single = eng.transform(X.iloc[[0]])
    idx = FEATURE_NAMES.index("rolling_src_std")
    assert single[0, idx] != 0.0
    assert single[0, idx] == pytest.approx(eng.rolling_std_fallback_)


def test_single_row_rolling_features_match_training_scale(fitted_engineer):
    """Serve-time rolling values must sit within the training distribution."""
    eng, X = fitted_engineer
    batch = eng.transform(X)
    single = eng.transform(X.iloc[[0]])

    for name in ROLLING_FEATURES:
        i = FEATURE_NAMES.index(name)
        train_mean = batch[:, i].mean()
        train_std = batch[:, i].std()
        z = abs(single[0, i] - train_mean) / (train_std + 1e-9)
        assert z < 3.0, f"{name} is {z:.1f} sigma from the training mean at serve time"


def test_non_rolling_features_are_row_independent(fitted_engineer):
    """Every non-rolling feature must be identical alone and inside a batch."""
    eng, X = fitted_engineer
    batch = eng.transform(X)
    single = eng.transform(X.iloc[[0]])

    for i, name in enumerate(FEATURE_NAMES):
        if name in ROLLING_FEATURES:
            continue
        assert single[0, i] == pytest.approx(batch[0, i]), (
            f"feature {name} changes value depending on batch size"
        )


def test_single_row_and_batch_anomaly_rates_agree(tmp_path):
    """Scoring rows one at a time must flag at the same rate as a batch.

    Under the skew bug the single-row rate went to ~100% while the batch rate
    stayed near the 5% contamination setting.
    """
    X, _ = generate_synthetic_data(300)
    pipe = train_anomaly_detector(X, model_path=str(tmp_path / "anom.joblib"))

    single_flags = [score_anomaly(X.iloc[[i]], pipe)["is_anomaly"] for i in range(100)]
    single_rate = float(np.mean(single_flags))
    batch_rate = batch_anomaly_rate(X.iloc[:100], pipe)

    assert single_rate == pytest.approx(batch_rate, abs=0.05), (
        f"single-row rate {single_rate:.2%} diverges from batch rate {batch_rate:.2%}"
    )


def test_in_distribution_traffic_is_mostly_not_flagged(tmp_path):
    """Contamination is 0.05, so in-distribution traffic must stay mostly clean."""
    X, _ = generate_synthetic_data(300)
    pipe = train_anomaly_detector(X, model_path=str(tmp_path / "anom2.joblib"))
    flags = [score_anomaly(X.iloc[[i]], pipe)["is_anomaly"] for i in range(100)]
    assert float(np.mean(flags)) < 0.20
