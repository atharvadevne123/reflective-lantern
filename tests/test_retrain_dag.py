"""Tests for retrain_dag pipeline helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_df(n: int = 200, mean: float = 10.0, std: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({"consumption_kwh": rng.normal(mean, std, n)})


def test_check_drift_no_train_file_returns_false(tmp_path) -> None:
    from pipelines.retrain_dag import check_drift_before_retrain

    # Use a path that definitely doesn't exist as the "train" path
    # by patching pathlib.Path inside the function's local scope
    fake_train = MagicMock()
    fake_train.exists.return_value = False
    ref = str(tmp_path / "ref.parquet")

    with patch("pathlib.Path") as MockPath:
        MockPath.side_effect = lambda p: fake_train if "wg_train" in str(p) else MagicMock()
        result = check_drift_before_retrain(reference_path=ref)
    assert result is False


def test_check_drift_no_reference_returns_true(tmp_path) -> None:
    from pipelines.retrain_dag import check_drift_before_retrain

    df_train = _make_df()
    ref_path = str(tmp_path / "ref.parquet")

    train_mock = MagicMock()
    train_mock.exists.return_value = True
    ref_mock = MagicMock()
    ref_mock.exists.return_value = False

    def path_factory(p):
        return train_mock if "wg_train" in str(p) else ref_mock

    with (
        patch("pathlib.Path", side_effect=path_factory),
        patch("pandas.read_parquet", return_value=df_train),
        patch("pandas.DataFrame.to_parquet"),
    ):
        result = check_drift_before_retrain(reference_path=ref_path)
    assert bool(result) is True


def test_check_drift_similar_distributions_returns_false(tmp_path) -> None:
    from pipelines.retrain_dag import check_drift_before_retrain

    df_same = _make_df(n=500, mean=10.0, std=1.0)
    ref_path = str(tmp_path / "ref.parquet")

    train_mock = MagicMock()
    train_mock.exists.return_value = True
    ref_mock = MagicMock()
    ref_mock.exists.return_value = True

    def path_factory(p):
        return train_mock if "wg_train" in str(p) else ref_mock

    with (
        patch("pathlib.Path", side_effect=path_factory),
        patch("pandas.read_parquet", return_value=df_same),
    ):
        result = check_drift_before_retrain(reference_path=ref_path)
    assert result == False  # noqa: E712 — np.False_ != False with `is`


def test_check_drift_different_distributions_returns_true(tmp_path) -> None:
    from pipelines.retrain_dag import check_drift_before_retrain

    df_ref = _make_df(n=500, mean=10.0, std=1.0)
    df_new = _make_df(n=500, mean=100.0, std=1.0)
    ref_path = str(tmp_path / "ref.parquet")

    train_mock = MagicMock()
    train_mock.exists.return_value = True
    ref_mock = MagicMock()
    ref_mock.exists.return_value = True

    call_count = [0]

    def fake_read_parquet(p, **kwargs):
        call_count[0] += 1
        return df_new if call_count[0] == 1 else df_ref

    def path_factory(p):
        return train_mock if "wg_train" in str(p) else ref_mock

    with (
        patch("pathlib.Path", side_effect=path_factory),
        patch("pandas.read_parquet", side_effect=fake_read_parquet),
        patch("pandas.DataFrame.to_parquet"),
    ):
        result = check_drift_before_retrain(reference_path=ref_path)
    assert bool(result) is True
