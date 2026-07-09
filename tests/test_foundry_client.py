"""Tests for scripts.foundry_client."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts.foundry_client import (
    FoundryAPIError,
    FoundryClient,
    FoundryConfigError,
    client_from_settings,
)


class FakeResponse:
    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self._body = json.dumps(body).encode() if body is not None else b""

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass


def make_client() -> FoundryClient:
    return FoundryClient("https://stack.palantirfoundry.com", "tok-123")


def test_client_requires_hostname_and_token() -> None:
    with pytest.raises(FoundryConfigError):
        FoundryClient("", "tok")
    with pytest.raises(FoundryConfigError):
        FoundryClient("https://stack.palantirfoundry.com", "")


def test_client_strips_trailing_slash() -> None:
    c = FoundryClient("https://stack.palantirfoundry.com/", "tok")
    assert c.base_url == "https://stack.palantirfoundry.com/api/v2"


def test_create_transaction_returns_rid() -> None:
    client = make_client()
    captured: dict[str, Any] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> FakeResponse:
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data or b"{}")
        return FakeResponse({"rid": "ri.foundry.main.transaction.abc"})

    with patch.object(urllib.request, "urlopen", fake_urlopen):
        rid = client.create_transaction("ri.foundry.main.dataset.x", branch="dev")

    assert rid == "ri.foundry.main.transaction.abc"
    assert captured["method"] == "POST"
    assert "branchName=dev" in captured["url"]
    assert captured["body"] == {"transactionType": "SNAPSHOT"}


def test_create_transaction_raises_without_rid() -> None:
    client = make_client()
    with patch.object(urllib.request, "urlopen", lambda req, timeout: FakeResponse({"nope": 1})):
        with pytest.raises(FoundryAPIError, match="No transaction RID"):
            client.create_transaction("ri.x")


def test_upload_file_hits_upload_endpoint() -> None:
    client = make_client()
    captured: dict[str, Any] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> FakeResponse:
        captured["url"] = req.full_url
        captured["content_type"] = req.get_header("Content-type")
        captured["data"] = req.data
        return FakeResponse()

    with patch.object(urllib.request, "urlopen", fake_urlopen):
        client.upload_file("ri.ds", "ri.txn", "runs.csv", b"a,b\n1,2\n")

    assert "/datasets/ri.ds/files/runs.csv/upload" in captured["url"]
    assert "transactionRid=ri.txn" in captured["url"]
    assert captured["content_type"] == "application/octet-stream"
    assert captured["data"] == b"a,b\n1,2\n"


def test_request_sends_bearer_token() -> None:
    client = make_client()
    captured: dict[str, Any] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> FakeResponse:
        captured["auth"] = req.get_header("Authorization")
        return FakeResponse({})

    with patch.object(urllib.request, "urlopen", fake_urlopen):
        client.commit_transaction("ri.ds", "ri.txn")

    assert captured["auth"] == "Bearer tok-123"


def test_upload_dataset_file_full_flow(tmp_path: Path) -> None:
    client = make_client()
    local = tmp_path / "runs.csv"
    local.write_text("a,b\n1,2\n")
    calls: list[str] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> FakeResponse:
        calls.append(req.full_url)
        if "/transactions?" in req.full_url:
            return FakeResponse({"rid": "ri.txn.1"})
        return FakeResponse()

    with patch.object(urllib.request, "urlopen", fake_urlopen):
        txn = client.upload_dataset_file("ri.ds", local, target_name="runs.csv")

    assert txn == "ri.txn.1"
    assert any("/upload" in u for u in calls)
    assert any(u.endswith("/commit") for u in calls)


def test_upload_dataset_file_aborts_on_failure(tmp_path: Path) -> None:
    client = make_client()
    local = tmp_path / "runs.csv"
    local.write_text("a,b\n")
    calls: list[str] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> FakeResponse:
        calls.append(req.full_url)
        if "/transactions?" in req.full_url:
            return FakeResponse({"rid": "ri.txn.9"})
        if "/upload" in req.full_url:
            raise urllib.error.URLError("connection reset")
        return FakeResponse()

    import urllib.error

    with patch.object(urllib.request, "urlopen", fake_urlopen):
        with pytest.raises(FoundryAPIError):
            client.upload_dataset_file("ri.ds", local)

    assert any(u.endswith("/abort") for u in calls)


def test_client_from_settings_unconfigured_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("FOUNDRY_HOSTNAME", "FOUNDRY_TOKEN", "FOUNDRY_DATASET_RID"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(FoundryConfigError):
        client_from_settings()


def test_client_from_settings_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "https://stack.palantirfoundry.com")
    monkeypatch.setenv("FOUNDRY_TOKEN", "tok")
    client = client_from_settings()
    assert client.base_url == "https://stack.palantirfoundry.com/api/v2"


def test_get_dataset_returns_metadata() -> None:
    client = make_client()
    with patch.object(
        urllib.request,
        "urlopen",
        lambda req, timeout: FakeResponse({"rid": "ri.ds", "name": "runs"}),
    ):
        meta = client.get_dataset("ri.ds")
    assert meta["name"] == "runs"


def test_list_files_returns_paths() -> None:
    client = make_client()
    body = {"data": [{"path": "runs.csv"}, {"path": "extra.json"}]}
    captured: dict[str, Any] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> FakeResponse:
        captured["url"] = req.full_url
        return FakeResponse(body)

    with patch.object(urllib.request, "urlopen", fake_urlopen):
        files = client.list_files("ri.ds", branch="dev")

    assert files == ["runs.csv", "extra.json"]
    assert "branchName=dev" in captured["url"]


def test_list_files_handles_malformed_response() -> None:
    client = make_client()
    with patch.object(
        urllib.request, "urlopen", lambda req, timeout: FakeResponse({"data": "oops"})
    ):
        assert client.list_files("ri.ds") == []


def test_main_verify_unconfigured_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from scripts.foundry_client import main

    for var in ("FOUNDRY_HOSTNAME", "FOUNDRY_TOKEN", "FOUNDRY_DATASET_RID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "argv", ["foundry_client.py", "--verify"])
    assert main() == 1


def test_main_requires_upload_or_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from scripts.foundry_client import main

    monkeypatch.setattr(sys, "argv", ["foundry_client.py"])
    with pytest.raises(SystemExit):
        main()


def test_foundry_client_raises_on_empty_hostname() -> None:
    from scripts.foundry_client import FoundryClient, FoundryConfigError

    with pytest.raises(FoundryConfigError):
        FoundryClient(hostname="", token="tok")


def test_foundry_client_raises_on_empty_token() -> None:
    from scripts.foundry_client import FoundryClient, FoundryConfigError

    with pytest.raises(FoundryConfigError):
        FoundryClient(hostname="https://host.example.com", token="")


def test_foundry_config_error_is_runtime_error() -> None:
    from scripts.foundry_client import FoundryConfigError

    assert issubclass(FoundryConfigError, RuntimeError)


def test_foundry_api_error_is_runtime_error() -> None:
    from scripts.foundry_client import FoundryAPIError

    assert issubclass(FoundryAPIError, RuntimeError)
