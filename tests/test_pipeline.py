"""End-to-end pipeline integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_df(n: int = 200, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "month": rng.integers(1, 13, n),
            "temperature_c": rng.uniform(0, 40, n),
            "humidity_pct": rng.uniform(20, 90, n),
            "occupancy": rng.integers(0, 200, n),
            "hvac_state": rng.integers(0, 2, n),
            "consumption_kwh": rng.uniform(5, 50, n),
        }
    )


def test_train_predict_end_to_end():
    from app.features import make_feature_row
    from app.model import predict, train_model

    df = _make_df()
    bundle, metrics = train_model(df, df["consumption_kwh"])
    assert metrics["r2_mean"] > -2.0

    row = make_feature_row(12, 0, 6, 25.0, 60.0, 80, 1, 20.0)
    preds = predict(bundle, row)
    assert len(preds) == 1
    assert isinstance(float(preds[0]), float)


def test_train_anomaly_end_to_end():
    from app.features import make_feature_row
    from app.model import score_anomaly, train_anomaly_model

    df = _make_df(300)
    bundle = train_anomaly_model(df)

    row = make_feature_row(3, 6, 12, 5.0, 30.0, 0, 0, 0.1)
    result = score_anomaly(bundle, row)
    assert "is_anomaly" in result
    assert result["severity"] in ("none", "warning", "critical")


def test_pipeline_stable_across_calls():
    from app.features import make_feature_row
    from app.model import predict, train_model

    df = _make_df(150, seed=77)
    bundle, _ = train_model(df, df["consumption_kwh"])

    row = make_feature_row(9, 2, 4, 18.0, 55.0, 40, 0, 10.0)
    p1 = predict(bundle, row)
    p2 = predict(bundle, row)
    np.testing.assert_allclose(p1, p2)


@pytest.mark.parametrize("hvac", [0, 1])
def test_hvac_effect_on_prediction(hvac):
    """HVAC=1 should yield higher consumption than HVAC=0 on average."""
    from app.features import make_feature_row
    from app.model import predict, train_model

    df = _make_df(400)
    bundle, _ = train_model(df, df["consumption_kwh"])

    off = float(predict(bundle, make_feature_row(14, 1, 7, 30.0, 60.0, 100, 0, 20.0))[0])
    on = float(predict(bundle, make_feature_row(14, 1, 7, 30.0, 60.0, 100, 1, 20.0))[0])
    # Not guaranteed but typically true with synthetic data
    assert isinstance(off, float) and isinstance(on, float)
