"""HMAC-verified inbound webhook processing.

Verifies the request signature before dispatching to registered event
handlers, preventing spoofed payloads from being acted upon.
"""

from __future__ import annotations

import hmac
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SignatureError",
    "WebhookEvent",
    "WebhookHandler",
]

logger = logging.getLogger(__name__)


class SignatureError(Exception):
    """Raised when a webhook signature cannot be verified."""


@dataclass
class WebhookEvent:
    """A parsed and verified webhook event."""

    event_type: str
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


EventHandler = Callable[[WebhookEvent], None]


class WebhookHandler:
    """Verifies and dispatches inbound webhook requests."""

    def __init__(self, secret: str, algorithm: str = "sha256") -> None:
        self._secret = secret.encode() if isinstance(secret, str) else secret
        self._algorithm = algorithm
        self._handlers: dict[str, list[EventHandler]] = {}
        self._catch_all: list[EventHandler] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Register *handler* for a specific event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    def on_any(self, handler: EventHandler) -> None:
        """Register *handler* for every event type."""
        self._catch_all.append(handler)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def verify_signature(self, body: bytes, signature: str) -> None:
        """Verify *signature* matches HMAC of *body*.

        Args:
            body: Raw request body bytes.
            signature: Header value like ``sha256=<hex>``.

        Raises:
            SignatureError: If verification fails.
        """
        try:
            _scheme, provided_digest = signature.split("=", 1)
        except ValueError as exc:
            raise SignatureError(f"Malformed signature: {signature!r}") from exc

        expected = hmac.new(self._secret, body, self._algorithm).hexdigest()
        if not hmac.compare_digest(expected, provided_digest):
            raise SignatureError("Signature mismatch")

    def process(
        self,
        body: bytes,
        event_type: str,
        signature: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> WebhookEvent:
        """Parse, verify, and dispatch a webhook payload.

        Args:
            body: Raw JSON request body.
            event_type: The event type string (e.g. ``push``, ``pr.opened``).
            signature: Optional HMAC signature header value.
            headers: Optional raw headers dict.

        Returns:
            The parsed :class:`WebhookEvent`.

        Raises:
            SignatureError: If *signature* is provided but invalid.
            ValueError: If *body* is not valid JSON.
        """
        if signature is not None:
            self.verify_signature(body, signature)

        payload = json.loads(body)
        event = WebhookEvent(
            event_type=event_type,
            payload=payload,
            headers=headers or {},
        )

        for handler in self._handlers.get(event_type, []):
            try:
                handler(event)
            except Exception:
                logger.exception("Handler error for event '%s'", event_type)

        for handler in self._catch_all:
            try:
                handler(event)
            except Exception:
                logger.exception("Catch-all handler error for event '%s'", event_type)

        return event
