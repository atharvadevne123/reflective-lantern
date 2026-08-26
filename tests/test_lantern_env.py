"""Tests for runtime capability detection and report delivery fallback."""

from __future__ import annotations

import json

import pytest

# Import via the rootdir that pytest already places on sys.path (tests/ is a
# package). Do NOT insert the repo root into sys.path here: this repository
# contains several sibling `app/` packages, and reordering sys.path mid-session
# changes which one later modules resolve to, which breaks unrelated suites.
from scripts import lantern_env, send_report


class TestCapabilities:
    """Tests for the Capabilities container."""

    def test_defaults_are_conservative(self):
        """An unprobed Capabilities assumes nothing is permitted."""
        caps = lantern_env.Capabilities()
        assert caps.credentials_honoured is False
        assert caps.can_enumerate_repos is False
        assert caps.can_create_repo is False
        assert caps.smtp_reachable is False

    def test_roundtrips_through_dict(self):
        """to_dict output can rebuild an equivalent Capabilities."""
        caps = lantern_env.Capabilities(can_push_git=True, notes=["ok"])
        rebuilt = lantern_env.Capabilities(**caps.to_dict())
        assert rebuilt.can_push_git is True
        assert rebuilt.notes == ["ok"]

    def test_summary_lists_every_capability(self):
        """The summary names each probed capability."""
        text = lantern_env.Capabilities(notes=["n"]).summary()
        for label in ("credentials honoured", "enumerate repos", "create repositories", "git push", "SMTP reachable"):
            assert label in text


class TestCredentialProbe:
    """Tests for the credentials_are_honoured probe."""

    def test_401_means_credentials_honoured(self, monkeypatch):
        """Real GitHub rejects an invalid token, proving tokens are used."""
        monkeypatch.setattr(lantern_env, "_request", lambda *a, **k: (401, ""))
        honoured, note = lantern_env.credentials_are_honoured()
        assert honoured is True
        assert "honoured" in note

    def test_200_means_credentials_ignored(self, monkeypatch):
        """A 200 for an invalid token proves the caller's token is discarded."""
        monkeypatch.setattr(lantern_env, "_request", lambda *a, **k: (200, "{}"))
        honoured, note = lantern_env.credentials_are_honoured()
        assert honoured is False
        assert "IGNORED" in note

    def test_unexpected_status_is_not_a_pass(self, monkeypatch):
        """An indeterminate status must not be reported as honoured."""
        monkeypatch.setattr(lantern_env, "_request", lambda *a, **k: (500, ""))
        honoured, _ = lantern_env.credentials_are_honoured()
        assert honoured is False


class TestEnumerationProbe:
    """Tests for the can_enumerate_repos probe."""

    def test_json_list_means_allowed(self, monkeypatch):
        """A JSON array response means enumeration works."""
        monkeypatch.setattr(lantern_env, "_request", lambda *a, **k: (200, json.dumps([{"name": "r"}])))
        assert lantern_env.can_enumerate_repos()[0] is True

    def test_session_binding_message_means_blocked(self, monkeypatch):
        """The proxy's session-binding message is recognised as a block."""
        body = json.dumps({"message": "sessions are bound to their configured repositories"})
        monkeypatch.setattr(lantern_env, "_request", lambda *a, **k: (200, body))
        allowed, note = lantern_env.can_enumerate_repos()
        assert allowed is False
        assert "BLOCKED" in note

    @pytest.mark.parametrize("status", [401, 403, 404, 0])
    def test_error_statuses_mean_blocked(self, monkeypatch, status):
        """Any non-200 status is treated as blocked."""
        monkeypatch.setattr(lantern_env, "_request", lambda *a, **k: (status, ""))
        assert lantern_env.can_enumerate_repos()[0] is False


class TestRepoCreationProbe:
    """Tests for the can_create_repo probe."""

    def test_422_means_creation_permitted(self, monkeypatch):
        """Reaching validation proves the caller may create repositories."""
        monkeypatch.setattr(lantern_env, "_request", lambda *a, **k: (422, ""))
        assert lantern_env.can_create_repo()[0] is True

    def test_403_means_forbidden(self, monkeypatch):
        """An installation token is rejected before validation."""
        monkeypatch.setattr(lantern_env, "_request", lambda *a, **k: (403, ""))
        allowed, note = lantern_env.can_create_repo()
        assert allowed is False
        assert "FORBIDDEN" in note

    def test_probe_sends_invalid_name_so_nothing_is_created(self, monkeypatch):
        """The probe must post an empty name so no repository can be created."""
        captured: dict = {}

        def fake(path, token=None, method="GET", payload=None):
            captured["path"] = path
            captured["method"] = method
            captured["payload"] = payload
            return 403, ""

        monkeypatch.setattr(lantern_env, "_request", fake)
        lantern_env.can_create_repo()
        assert captured["method"] == "POST"
        assert captured["payload"] == {"name": ""}, "probe must not send a usable name"


class TestSmtpProbe:
    """Tests for the smtp_reachable probe."""

    def test_unreachable_when_connection_fails(self, monkeypatch):
        """All ports failing yields an unreachable verdict."""

        def boom(*a, **k):
            raise OSError("blocked")

        monkeypatch.setattr(lantern_env.socket, "create_connection", boom)
        reachable, note = lantern_env.smtp_reachable()
        assert reachable is False
        assert "UNREACHABLE" in note


class TestDeliveryFallback:
    """Tests for report delivery degrading to a repository file."""

    def test_skips_email_without_credentials(self, monkeypatch):
        """Email is skipped when no password is configured."""
        monkeypatch.delenv("LANTERN_SMTP_PASSWORD", raising=False)
        monkeypatch.delenv("LANTERN_REPORT_TO", raising=False)
        assert send_report.send_email("s", "b") is False

    def test_deliver_falls_back_to_file(self, monkeypatch, tmp_path):
        """When email is unavailable the report is still written to disk."""
        monkeypatch.setattr(send_report, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(send_report, "send_email", lambda *a, **k: False)
        status = send_report.deliver("Run Report", "body text")
        assert status.startswith("filed:")
        assert list(tmp_path.glob("*.txt")), "no report file was written"

    def test_deliver_survives_email_exception(self, monkeypatch, tmp_path):
        """An exception during send must not fail the run."""

        def boom(*a, **k):
            raise RuntimeError("smtp exploded")

        monkeypatch.setattr(send_report, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(send_report, "send_email", boom)
        status = send_report.deliver("Run Report", "body")
        assert status.startswith("filed:")

    def test_filed_report_contains_body(self, monkeypatch, tmp_path):
        """The filed report preserves the report text."""
        monkeypatch.setattr(send_report, "REPORTS_DIR", tmp_path)
        path = send_report.file_to_repo("Subject Here", "the body")
        assert "the body" in path.read_text()
