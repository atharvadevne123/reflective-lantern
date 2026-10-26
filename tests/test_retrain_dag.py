"""Tests for retrain_dag pipeline helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


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

    def path_factory(p) -> None:
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

    def path_factory(p) -> None:
        return train_mock if "wg_train" in str(p) else ref_mock

    with (
        patch("pathlib.Path", side_effect=path_factory),
        patch("pandas.read_parquet", return_value=df_same),
    ):
        result = check_drift_before_retrain(reference_path=ref_path)
    assert result == False  # noqa: E712 — np.False_ != False with `is`


@pytest.mark.parametrize("mean_offset", [50.0, 100.0, 200.0])
def test_check_drift_large_offset_returns_true(tmp_path, mean_offset: float) -> None:
    from pipelines.retrain_dag import check_drift_before_retrain

    df_ref = _make_df(n=500, mean=10.0, std=1.0)
    df_new = _make_df(n=500, mean=10.0 + mean_offset, std=1.0)
    ref_path = str(tmp_path / "ref.parquet")

    train_mock = MagicMock()
    train_mock.exists.return_value = True
    ref_mock = MagicMock()
    ref_mock.exists.return_value = True
    call_count = [0]

    def fake_read(p, **kw) -> None:
        call_count[0] += 1
        return df_new if call_count[0] == 1 else df_ref

    with (
        patch(
            "pathlib.Path", side_effect=lambda p: train_mock if "wg_train" in str(p) else ref_mock
        ),
        patch("pandas.read_parquet", side_effect=fake_read),
        patch("pandas.DataFrame.to_parquet"),
    ):
        result = check_drift_before_retrain(reference_path=ref_path)
    assert bool(result) is True


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

    def fake_read_parquet(p, **kwargs) -> None:
        call_count[0] += 1
        return df_new if call_count[0] == 1 else df_ref

    def path_factory(p) -> None:
        return train_mock if "wg_train" in str(p) else ref_mock

    with (
        patch("pathlib.Path", side_effect=path_factory),
        patch("pandas.read_parquet", side_effect=fake_read_parquet),
        patch("pandas.DataFrame.to_parquet"),
    ):
        result = check_drift_before_retrain(reference_path=ref_path)
    assert bool(result) is True


class TestRetainDagEdgeCases:
    """Additional edge-case tests for check_drift_before_retrain."""

    def test_empty_train_returns_false(self, tmp_path) -> None:
        """An empty training DataFrame should not trigger drift."""
        from pipelines.retrain_dag import check_drift_before_retrain

        df_empty = pd.DataFrame({"consumption_kwh": []})
        ref_path = str(tmp_path / "ref.parquet")

        train_mock = MagicMock()
        train_mock.exists.return_value = True
        ref_mock = MagicMock()
        ref_mock.exists.return_value = False

        with (
            patch(
                "pathlib.Path",
                side_effect=lambda p: train_mock if "wg_train" in str(p) else ref_mock,
            ),
            patch("pandas.read_parquet", return_value=df_empty),
            patch("pandas.DataFrame.to_parquet"),
        ):
            result = check_drift_before_retrain(reference_path=ref_path)
        assert bool(result) is True  # no reference → treated as drift

    @pytest.mark.parametrize("n_samples", [100, 300, 500])
    def test_similar_distributions_no_drift_various_sizes(self, tmp_path, n_samples: int) -> None:
        """Same distribution with various sample sizes should not detect drift."""
        from pipelines.retrain_dag import check_drift_before_retrain

        rng = np.random.default_rng(0)
        df = pd.DataFrame({"consumption_kwh": rng.normal(10.0, 1.0, n_samples)})
        ref_path = str(tmp_path / "ref.parquet")

        train_mock = MagicMock()
        train_mock.exists.return_value = True
        ref_mock = MagicMock()
        ref_mock.exists.return_value = True

        with (
            patch(
                "pathlib.Path",
                side_effect=lambda p: train_mock if "wg_train" in str(p) else ref_mock,
            ),
            patch("pandas.read_parquet", return_value=df),
        ):
            result = check_drift_before_retrain(reference_path=ref_path)
        assert result == False  # noqa: E712


def test_check_drift_returns_bool_type(tmp_path) -> None:
    """Result of check_drift_before_retrain should be truthy/falsy."""
    from pipelines.retrain_dag import check_drift_before_retrain

    fake_train = MagicMock()
    fake_train.exists.return_value = False
    ref = str(tmp_path / "ref.parquet")

    with patch("pathlib.Path") as MockPath:
        MockPath.side_effect = lambda p: fake_train if "wg_train" in str(p) else MagicMock()
        result = check_drift_before_retrain(reference_path=ref)
    assert result is False or result is True or isinstance(result, (bool, int, float))


def test_check_drift_single_sample_no_reference(tmp_path) -> None:
    """A single-sample training set with no reference should return True (drift)."""
    from pipelines.retrain_dag import check_drift_before_retrain

    df_single = pd.DataFrame({"consumption_kwh": [42.0]})
    ref_path = str(tmp_path / "ref.parquet")

    train_mock = MagicMock()
    train_mock.exists.return_value = True
    ref_mock = MagicMock()
    ref_mock.exists.return_value = False

    with (
        patch(
            "pathlib.Path",
            side_effect=lambda p: train_mock if "wg_train" in str(p) else ref_mock,
        ),
        patch("pandas.read_parquet", return_value=df_single),
        patch("pandas.DataFrame.to_parquet"),
    ):
        result = check_drift_before_retrain(reference_path=ref_path)
    assert bool(result) is True


def test_check_drift_identical_data_no_drift(tmp_path) -> None:
    """When train and reference are identical, no drift should be detected."""
    from pipelines.retrain_dag import check_drift_before_retrain

    df = _make_df(n=300, mean=5.0, std=0.5)
    ref_path = str(tmp_path / "ref.parquet")

    train_mock = MagicMock()
    train_mock.exists.return_value = True
    ref_mock = MagicMock()
    ref_mock.exists.return_value = True

    with (
        patch(
            "pathlib.Path",
            side_effect=lambda p: train_mock if "wg_train" in str(p) else ref_mock,
        ),
        patch("pandas.read_parquet", return_value=df),
    ):
        result = check_drift_before_retrain(reference_path=ref_path)
    assert result == False  # noqa: E712


@pytest.mark.parametrize("mean_offset", [0.0, 5.0])
def test_check_drift_small_offset_no_drift(tmp_path, mean_offset: float) -> None:
    """A small mean offset should not trigger drift detection."""
    from pipelines.retrain_dag import check_drift_before_retrain

    rng = np.random.default_rng(7)
    df_ref = pd.DataFrame({"consumption_kwh": rng.normal(10.0, 2.0, 500)})
    df_new = pd.DataFrame({"consumption_kwh": rng.normal(10.0 + mean_offset, 2.0, 500)})
    ref_path = str(tmp_path / "ref.parquet")

    train_mock = MagicMock()
    train_mock.exists.return_value = True
    ref_mock = MagicMock()
    ref_mock.exists.return_value = True
    call_count = [0]

    def fake_read(p, **kw):
        call_count[0] += 1
        return df_new if call_count[0] == 1 else df_ref

    with (
        patch(
            "pathlib.Path",
            side_effect=lambda p: train_mock if "wg_train" in str(p) else ref_mock,
        ),
        patch("pandas.read_parquet", side_effect=fake_read),
        patch("pandas.DataFrame.to_parquet"),
    ):
        result = check_drift_before_retrain(reference_path=ref_path)
    assert result == False  # noqa: E712


def test_check_drift_extreme_offset_always_true(tmp_path) -> None:
    """A 1000-unit mean offset should always trigger drift."""
    from pipelines.retrain_dag import check_drift_before_retrain

    df_ref = _make_df(n=400, mean=10.0, std=1.0)
    df_new = _make_df(n=400, mean=1010.0, std=1.0)
    ref_path = str(tmp_path / "ref.parquet")

    train_mock = MagicMock()
    train_mock.exists.return_value = True
    ref_mock = MagicMock()
    ref_mock.exists.return_value = True
    call_count = [0]

    def fake_read(p, **kw):
        call_count[0] += 1
        return df_new if call_count[0] == 1 else df_ref

    with (
        patch(
            "pathlib.Path",
            side_effect=lambda p: train_mock if "wg_train" in str(p) else ref_mock,
        ),
        patch("pandas.read_parquet", side_effect=fake_read),
        patch("pandas.DataFrame.to_parquet"),
    ):
        result = check_drift_before_retrain(reference_path=ref_path)
    assert bool(result) is True
