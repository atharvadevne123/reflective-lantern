"""Tests for model training, prediction, and data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.model import generate_sample_data, predict_demand


class TestGenerateSampleData:
    def test_returns_dataframe_and_array(self):
        df, y = generate_sample_data(100)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(y, np.ndarray)

    def test_row_count_matches_n(self):
        df, y = generate_sample_data(50)
        assert len(df) == 50
        assert len(y) == 50

    def test_target_is_binary(self):
        _, y = generate_sample_data(200)
        assert set(np.unique(y)).issubset({0, 1})

    def test_required_columns_present(self):
        df, _ = generate_sample_data(10)
        required = {
            "lead_time",
            "length_of_stay",
            "guests_count",
            "checkin_month",
            "checkin_dayofweek",
            "is_weekend",
            "current_occ_rate",
            "prev_year_occ_rate",
            "room_rate",
            "competitor_avg_rate",
            "special_event",
            "room_type",
            "booking_channel",
        }
        assert required.issubset(set(df.columns))

    def test_room_rate_is_positive(self):
        df, _ = generate_sample_data(50)
        assert (df["room_rate"] > 0).all()

    def test_occ_rate_in_valid_range(self):
        df, _ = generate_sample_data(100)
        assert df["current_occ_rate"].between(0, 1).all()

    @pytest.mark.parametrize("n", [10, 100, 500])
    def test_reproducible_with_fixed_seed(self, n):
        _, y1 = generate_sample_data(n)
        _, y2 = generate_sample_data(n)
        np.testing.assert_array_equal(y1, y2)


class TestTrainModel:
    def test_returns_three_items(self, trained_models):
        assert len(trained_models) == 3

    def test_metrics_has_auc_mean(self, trained_models):
        _, _, metrics = trained_models
        assert "auc_mean" in metrics
        assert 0.0 <= float(metrics["auc_mean"]) <= 1.0

    def test_auc_mean_above_random(self, trained_models):
        _, _, metrics = trained_models
        assert float(metrics["auc_mean"]) > 0.55

    def test_n_features_in_metrics(self, trained_models):
        _, _, metrics = trained_models
        assert metrics["n_features"] == 18

    def test_both_models_fit(self, trained_models):
        xgb_pipe, lgbm_pipe, _ = trained_models
        assert xgb_pipe is not None
        assert lgbm_pipe is not None

    def test_xgb_can_predict_proba(self, trained_models, sample_booking_df):
        xgb_pipe, _, _ = trained_models
        probs = xgb_pipe.predict_proba(sample_booking_df)
        assert probs.shape[1] == 2
        assert 0.0 <= probs[0, 1] <= 1.0


class TestPredictDemand:
    def test_demand_score_in_range(self, trained_models, sample_booking_df):
        xgb_pipe, lgbm_pipe, _ = trained_models
        result = predict_demand(xgb_pipe, lgbm_pipe, sample_booking_df)
        assert 0.0 <= result["demand_score"] <= 1.0

    def test_suggested_rate_is_positive(self, trained_models, sample_booking_df):
        xgb_pipe, lgbm_pipe, _ = trained_models
        result = predict_demand(xgb_pipe, lgbm_pipe, sample_booking_df, base_rate=150.0)
        assert result["suggested_rate"] > 0

    def test_demand_tier_is_valid(self, trained_models, sample_booking_df):
        xgb_pipe, lgbm_pipe, _ = trained_models
        result = predict_demand(xgb_pipe, lgbm_pipe, sample_booking_df)
        assert result["demand_tier"] in {"low", "medium", "high"}

    def test_per_row_scores_length(self, trained_models, sample_booking_df):
        xgb_pipe, lgbm_pipe, _ = trained_models
        result = predict_demand(xgb_pipe, lgbm_pipe, sample_booking_df)
        assert len(result["per_row_scores"]) == len(sample_booking_df)

    def test_works_without_lgbm(self, trained_models, sample_booking_df):
        xgb_pipe, _, _ = trained_models
        result = predict_demand(xgb_pipe, None, sample_booking_df)
        assert "demand_score" in result

    @pytest.mark.parametrize("base_rate", [80.0, 150.0, 300.0])
    def test_suggested_rate_scales_with_base(self, trained_models, sample_booking_df, base_rate):
        xgb_pipe, lgbm_pipe, _ = trained_models
        result = predict_demand(xgb_pipe, lgbm_pipe, sample_booking_df, base_rate=base_rate)
        # Rate must be in [0.7×base, 1.6×base]
        assert result["suggested_rate"] >= base_rate * 0.69
        assert result["suggested_rate"] <= base_rate * 1.61

    def test_high_demand_gives_high_tier(self, trained_models):
        xgb_pipe, lgbm_pipe, _ = trained_models
        from unittest.mock import patch

        import numpy as np

        high_probs = np.array([0.85])
        with (
            patch.object(
                xgb_pipe,
                "predict_proba",
                return_value=np.column_stack([1 - high_probs, high_probs]),
            ),
            patch.object(
                lgbm_pipe,
                "predict_proba",
                return_value=np.column_stack([1 - high_probs, high_probs]),
            ),
        ):
            X_dummy = pd.DataFrame([{"a": 1}])
            result = predict_demand(xgb_pipe, lgbm_pipe, X_dummy)
        assert result["demand_tier"] == "high"

    def test_low_demand_gives_low_tier(self, trained_models):
        xgb_pipe, lgbm_pipe, _ = trained_models
        from unittest.mock import patch

        import numpy as np

        low_probs = np.array([0.20])
        with (
            patch.object(
                xgb_pipe, "predict_proba", return_value=np.column_stack([1 - low_probs, low_probs])
            ),
            patch.object(
                lgbm_pipe, "predict_proba", return_value=np.column_stack([1 - low_probs, low_probs])
            ),
        ):
            X_dummy = pd.DataFrame([{"a": 1}])
            result = predict_demand(xgb_pipe, lgbm_pipe, X_dummy)
        assert result["demand_tier"] == "low"


@pytest.mark.parametrize("n", [50, 100, 500])
def test_generate_sample_data_row_count(n: int) -> None:
    from app.model import generate_sample_data

    df, y = generate_sample_data(n)
    assert len(df) == n
    assert len(y) == n


@pytest.mark.parametrize(
    "key",
    ["demand_score", "demand_tier", "confidence"],
)
def test_predict_demand_output_keys(key: str) -> None:
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    from app.model import _make_lgbm_pipe, _make_xgb_pipe, predict_demand

    xgb_pipe = _make_xgb_pipe()
    lgbm_pipe = _make_lgbm_pipe()
    probs = np.array([0.7])
    with (
        patch.object(xgb_pipe, "predict_proba", return_value=np.column_stack([1 - probs, probs])),
        patch.object(lgbm_pipe, "predict_proba", return_value=np.column_stack([1 - probs, probs])),
    ):
        result = predict_demand(xgb_pipe, lgbm_pipe, pd.DataFrame([{"a": 1}]))
    assert key in result


@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
def test_predict_demand_score_in_unit_interval(prob: float) -> None:
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    from app.model import _make_lgbm_pipe, _make_xgb_pipe, predict_demand

    xgb_pipe = _make_xgb_pipe()
    lgbm_pipe = _make_lgbm_pipe()
    probs = np.array([prob])
    with (
        patch.object(xgb_pipe, "predict_proba", return_value=np.column_stack([1 - probs, probs])),
        patch.object(lgbm_pipe, "predict_proba", return_value=np.column_stack([1 - probs, probs])),
    ):
        result = predict_demand(xgb_pipe, lgbm_pipe, pd.DataFrame([{"a": 1}]))
    assert 0.0 <= result["demand_score"] <= 1.0


def test_predict_demand_tier_is_string() -> None:
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    from app.model import _make_lgbm_pipe, _make_xgb_pipe, predict_demand

    xgb_pipe = _make_xgb_pipe()
    lgbm_pipe = _make_lgbm_pipe()
    probs = np.array([0.6])
    with (
        patch.object(xgb_pipe, "predict_proba", return_value=np.column_stack([1 - probs, probs])),
        patch.object(lgbm_pipe, "predict_proba", return_value=np.column_stack([1 - probs, probs])),
    ):
        result = predict_demand(xgb_pipe, lgbm_pipe, pd.DataFrame([{"a": 1}]))
    assert isinstance(result["demand_tier"], str)
