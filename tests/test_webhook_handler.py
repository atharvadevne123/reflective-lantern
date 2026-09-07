"""Tests for app.webhook_handler."""

import hashlib
import hmac
import json

import pytest

from app.webhook_handler import SignatureError, WebhookEvent, WebhookHandler

SECRET = "test-secret"


def make_sig(body: bytes, secret: str = SECRET) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class TestVerifySignature:
    def test_valid_signature(self):
        handler = WebhookHandler(SECRET)
        body = b'{"a": 1}'
        sig = make_sig(body)
        handler.verify_signature(body, sig)  # should not raise

    def test_invalid_signature_raises(self):
        handler = WebhookHandler(SECRET)
        body = b'{"a": 1}'
        with pytest.raises(SignatureError):
            handler.verify_signature(body, "sha256=deadbeef")

    def test_malformed_signature_raises(self):
        handler = WebhookHandler(SECRET)
        with pytest.raises(SignatureError):
            handler.verify_signature(b"body", "noseparator")


class TestProcess:
    def test_dispatches_to_handler(self):
        wh = WebhookHandler(SECRET)
        received = []
        wh.on("push", received.append)
        body = json.dumps({"ref": "main"}).encode()
        event = wh.process(body, "push", signature=make_sig(body))
        assert isinstance(event, WebhookEvent)
        assert len(received) == 1
        assert received[0].payload["ref"] == "main"

    def test_catch_all_handler(self):
        wh = WebhookHandler(SECRET)
        received = []
        wh.on_any(received.append)
        body = b'{"x": 1}'
        wh.process(body, "any_event", signature=make_sig(body))
        assert len(received) == 1

    def test_no_signature_skips_verification(self):
        wh = WebhookHandler(SECRET)
        body = b'{"k": "v"}'
        event = wh.process(body, "test")
        assert event.payload["k"] == "v"

    def test_wrong_signature_raises(self):
        wh = WebhookHandler(SECRET)
        body = b'{"k": "v"}'
        with pytest.raises(SignatureError):
            wh.process(body, "test", signature="sha256=wrong")

    def test_handler_exception_does_not_propagate(self):
        wh = WebhookHandler(SECRET)
        wh.on("ev", lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
        body = b'{"a": 1}'
        # Should not raise
        event = wh.process(body, "ev", signature=make_sig(body))
        assert event.event_type == "ev"

    def test_multiple_handlers_same_event(self):
        wh = WebhookHandler(SECRET)
        calls = []
        wh.on("ping", lambda e: calls.append(1))
        wh.on("ping", lambda e: calls.append(2))
        body = b'{"ping": true}'
        wh.process(body, "ping", signature=make_sig(body))
        assert calls == [1, 2]


class TestWebhookParametrized:
    @pytest.mark.parametrize("event_type", ["push", "pull_request", "release", "ping"])
    def test_event_type_stored(self, event_type: str) -> None:
        wh = WebhookHandler(SECRET)
        body = json.dumps({"type": event_type}).encode()
        event = wh.process(body, event_type, signature=make_sig(body))
        assert event.event_type == event_type

    @pytest.mark.parametrize("secret", ["abc", "my-secret", "s3cr3t!"])
    def test_signature_matches_secret(self, secret: str) -> None:
        wh = WebhookHandler(secret)
        body = b'{"ok": true}'
        sig = make_sig(body, secret)
        event = wh.process(body, "push", signature=sig)
        assert event.payload["ok"] is True

    @pytest.mark.parametrize("n_handlers", [1, 2, 3])
    def test_catch_all_receives_all_events(self, n_handlers: int) -> None:
        wh = WebhookHandler(SECRET)
        calls: list[str] = []
        for _ in range(n_handlers):
            wh.on_any(lambda e: calls.append(e.event_type))
        body = b"{}"
        wh.process(body, "tick", signature=make_sig(body))
        assert len(calls) == n_handlers
