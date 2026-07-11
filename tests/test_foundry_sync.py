"""Tests for scripts.foundry_sync."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from config.settings import Settings
from scripts.foundry_sync import sync


def _unconfigured_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for var in ("FOUNDRY_HOSTNAME", "FOUNDRY_TOKEN", "FOUNDRY_DATASET_RID"):
        monkeypatch.delenv(var, raising=False)
    return Settings()


def _configured_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "https://stack.palantirfoundry.com")
    monkeypatch.setenv("FOUNDRY_TOKEN", "tok")
    monkeypatch.setenv("FOUNDRY_DATASET_RID", "ri.foundry.main.dataset.x")
    return Settings()


def test_sync_unconfigured_exports_without_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = sync(settings=_unconfigured_settings(monkeypatch))
    assert summary["uploaded"] is False
    assert summary["transaction_rid"] is None
    assert isinstance(summary["rows"], int) and summary["rows"] > 0


def test_sync_export_only_never_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _configured_settings(monkeypatch)
    with patch("scripts.foundry_sync.client_from_settings") as mocked:
        summary = sync(export_only=True, settings=settings)
    mocked.assert_not_called()
    assert summary["uploaded"] is False


def test_sync_configured_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _configured_settings(monkeypatch)
    with patch("scripts.foundry_sync.client_from_settings") as mocked:
        mocked.return_value.upload_dataset_files.return_value = "ri.txn.42"
        mocked.return_value.list_files.return_value = []
        summary = sync(settings=settings)
    assert summary["uploaded"] is True
    assert summary["transaction_rid"] == "ri.txn.42"
    call = mocked.return_value.upload_dataset_files.call_args
    assert call.args[0] == "ri.foundry.main.dataset.x"


def test_sync_jsonl_format_target_name(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _configured_settings(monkeypatch)
    with patch("scripts.foundry_sync.client_from_settings") as mocked:
        mocked.return_value.upload_dataset_files.return_value = "ri.txn.1"
        mocked.return_value.list_files.return_value = []
        summary = sync(fmt="jsonl", settings=settings)
    assert summary["format"] == "jsonl"
    files_arg = mocked.return_value.upload_dataset_files.call_args.args[1]
    assert any(name.endswith(".jsonl") for name in files_arg)


def test_sync_summary_row_count_matches_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.foundry_export import build_run_rows

    summary = sync(settings=_unconfigured_settings(monkeypatch))
    assert summary["rows"] == len(build_run_rows())


def test_sync_summary_has_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = sync(settings=_unconfigured_settings(monkeypatch))
    for key in ("rows", "format", "uploaded", "transaction_rid"):
        assert key in summary, f"Missing key: {key}"


def test_sync_format_defaults_to_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = sync(settings=_unconfigured_settings(monkeypatch))
    assert summary["format"] == "csv"


def test_sync_uploaded_false_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _unconfigured_settings(monkeypatch)
    assert not settings.foundry_configured()
    summary = sync(settings=settings)
    assert summary["uploaded"] is False


def test_sync_csv_format_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = sync(fmt="csv", settings=_unconfigured_settings(monkeypatch))
    assert summary["format"] == "csv"


def test_sync_uploads_csv_ontology_and_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _configured_settings(monkeypatch)
    with patch("scripts.foundry_sync.client_from_settings") as mocked:
        mocked.return_value.upload_dataset_files.return_value = "ri.txn.3"
        mocked.return_value.list_files.return_value = [
            "reflective_lantern_runs.csv",
            "reflective_lantern_ontology.json",
            "manifest.json",
        ]
        summary = sync(settings=settings)
    files_arg = mocked.return_value.upload_dataset_files.call_args.args[1]
    assert set(files_arg) == {
        "reflective_lantern_runs.csv",
        "reflective_lantern_ontology.json",
        "manifest.json",
    }
    assert summary["files"] == [
        "reflective_lantern_runs.csv",
        "reflective_lantern_ontology.json",
        "manifest.json",
    ]


def test_sync_no_ontology_skips_json(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _configured_settings(monkeypatch)
    with patch("scripts.foundry_sync.client_from_settings") as mocked:
        mocked.return_value.upload_dataset_files.return_value = "ri.txn.4"
        mocked.return_value.list_files.return_value = []
        sync(include_ontology=False, settings=settings)
    files_arg = mocked.return_value.upload_dataset_files.call_args.args[1]
    assert "reflective_lantern_ontology.json" not in files_arg
    assert "manifest.json" in files_arg


def test_build_manifest_contents() -> None:
    from scripts.foundry_sync import build_manifest

    m = build_manifest(10, "csv", True)
    assert m["rows"] == 10
    assert m["format"] == "csv"
    assert m["includes_ontology"] is True
    assert m["generator"] == "reflective-lantern"


def test_sync_manifest_is_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    settings = _configured_settings(monkeypatch)
    with patch("scripts.foundry_sync.client_from_settings") as mocked:
        mocked.return_value.upload_dataset_files.return_value = "ri.txn.5"
        mocked.return_value.list_files.return_value = []
        sync(settings=settings)
    files_arg = mocked.return_value.upload_dataset_files.call_args.args[1]
    manifest = _json.loads(files_arg["manifest.json"])
    assert manifest["rows"] > 0
