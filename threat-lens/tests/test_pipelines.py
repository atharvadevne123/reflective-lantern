"""Tests for the retraining pipeline."""

from unittest.mock import MagicMock, patch

from pipelines.retrain_dag import (
    DEFAULT_ARGS,
    MIN_NEW_SAMPLES,
    _load_reference_window,
    collect_new_samples,
    run_retraining_pipeline,
)


def test_reference_window_has_expected_features() -> None:
    ref = _load_reference_window()
    assert {"src_bytes", "dst_bytes", "duration", "confidence"} <= set(ref)
    assert all(len(v) > 0 for v in ref.values())


def test_default_args_configured() -> None:
    assert DEFAULT_ARGS["owner"] == "threat-lens"
    assert DEFAULT_ARGS["retries"] >= 1


def test_collect_new_samples_counts_rows() -> None:
    session = MagicMock()
    session.query.return_value.count.return_value = 42
    with patch("app.database.SessionLocal", return_value=session):
        assert collect_new_samples() == 42
    session.close.assert_called_once()


def test_pipeline_skips_retrain_when_quiet() -> None:
    """No drift and too few samples means no retraining run."""
    with (
        patch("pipelines.retrain_dag.check_drift", return_value=False),
        patch("pipelines.retrain_dag.collect_new_samples", return_value=0),
        patch("pipelines.retrain_dag.retrain_model") as retrain,
    ):
        run_retraining_pipeline()
    retrain.assert_not_called()


def test_pipeline_retrains_on_drift() -> None:
    with (
        patch("pipelines.retrain_dag.check_drift", return_value=True),
        patch("pipelines.retrain_dag.collect_new_samples", return_value=0),
        patch("pipelines.retrain_dag.retrain_model", return_value={}) as retrain,
    ):
        run_retraining_pipeline()
    retrain.assert_called_once()


def test_pipeline_retrains_on_sample_threshold() -> None:
    with (
        patch("pipelines.retrain_dag.check_drift", return_value=False),
        patch("pipelines.retrain_dag.collect_new_samples", return_value=MIN_NEW_SAMPLES),
        patch("pipelines.retrain_dag.retrain_model", return_value={}) as retrain,
    ):
        run_retraining_pipeline()
    retrain.assert_called_once()
