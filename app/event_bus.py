"""Lightweight synchronous event bus for decoupled internal messaging."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

Handler = Callable[[str, Any], None]


class EventBus:
    """Thread-unsafe synchronous pub/sub bus.

    Listeners are called in registration order.  Exceptions inside a
    listener are logged and do not prevent other listeners from running.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._wildcard_handlers: list[Handler] = []

    def subscribe(self, event: str, handler: Handler) -> None:
        """Register a handler for a named event.

        Args:
            event: Event name to listen for.  Use ``"*"`` for all events.
            handler: Callable receiving ``(event_name, payload)``.
        """
        if event == "*":
            self._wildcard_handlers.append(handler)
        else:
            self._handlers[event].append(handler)
        logger.debug("Subscribed %s to event '%s'", handler, event)

    def unsubscribe(self, event: str, handler: Handler) -> bool:
        """Remove a previously registered handler.

        Args:
            event: Event name or ``"*"`` for wildcard.
            handler: The exact callable to remove.

        Returns:
            True if the handler was found and removed, False otherwise.
        """
        if event == "*":
            try:
                self._wildcard_handlers.remove(handler)
                return True
            except ValueError:
                return False
        try:
            self._handlers[event].remove(handler)
            return True
        except ValueError:
            return False

    def publish(self, event: str, payload: Any = None) -> int:
        """Publish an event to all registered handlers.

        Args:
            event: Event name.
            payload: Arbitrary data passed to each handler.

        Returns:
            Number of handlers that were called.
        """
        called = 0
        handlers = list(self._handlers.get(event, [])) + list(self._wildcard_handlers)
        for handler in handlers:
            try:
                handler(event, payload)
                called += 1
            except Exception as exc:
                logger.error("Handler %s raised for event '%s': %s", handler, event, exc)
        logger.debug("Published '%s' to %d handler(s)", event, called)
        return called

    def clear(self, event: str | None = None) -> None:
        """Remove all handlers for an event, or all events if event is None.

        Args:
            event: Event name to clear, or None to clear everything.
        """
        if event is None:
            self._handlers.clear()
            self._wildcard_handlers.clear()
        elif event == "*":
            self._wildcard_handlers.clear()
        else:
            self._handlers.pop(event, None)

    def listener_count(self, event: str) -> int:
        """Return the number of listeners registered for an event.

        Args:
            event: Event name.

        Returns:
            Listener count for the specific event (excludes wildcards).
        """
        return len(self._handlers.get(event, []))

    def event_names(self) -> list[str]:
        """Return sorted list of event names that have at least one handler."""
        return sorted(k for k, v in self._handlers.items() if v)

    def total_listeners(self) -> int:
        """Return the total number of specific and wildcard handlers registered."""
        return sum(len(v) for v in self._handlers.values()) + len(self._wildcard_handlers)

    def has_listeners(self, event: str) -> bool:
        """Return True if any handlers are registered for *event*."""
        return bool(self._handlers.get(event)) or bool(self._wildcard_handlers)


_default_bus: EventBus = EventBus()


def get_bus() -> EventBus:
    """Return the process-wide default event bus."""
    return _default_bus


__all__ = ["EventBus", "Handler", "get_bus"]
