"""Multi-channel notification dispatcher.

Routes notifications to one or more registered channels (e.g. email,
Slack, webhook) based on severity and channel availability.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "Channel",
    "Notification",
    "NotificationDispatcher",
    "Severity",
]

logger = logging.getLogger(__name__)


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Notification:
    """A notification payload."""

    title: str
    body: str
    severity: Severity = Severity.INFO
    tags: list[str] = field(default_factory=list)


SendFn = Callable[[Notification], None]


@dataclass
class Channel:
    """A named delivery channel with a minimum severity filter."""

    name: str
    send: SendFn
    min_severity: Severity = Severity.INFO
    enabled: bool = True


_SEVERITY_ORDER = [Severity.INFO, Severity.WARNING, Severity.ERROR, Severity.CRITICAL]


class NotificationDispatcher:
    """Dispatches notifications to all eligible registered channels."""

    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}

    def register(self, channel: Channel) -> None:
        """Add or replace a channel by name.

        Args:
            channel: :class:`Channel` instance to register; overwrites any
                existing channel with the same :attr:`~Channel.name`.
        """
        self._channels[channel.name] = channel

    def unregister(self, name: str) -> None:
        """Remove a channel by name.

        Args:
            name: Name of the channel to remove. No-ops if not registered.
        """
        self._channels.pop(name, None)

    def dispatch(self, notification: Notification) -> dict[str, bool]:
        """Send *notification* to all eligible channels.

        A channel is eligible when it is enabled and its minimum severity is
        at or below the notification's severity.

        Args:
            notification: The :class:`Notification` to deliver.

        Returns:
            Mapping of channel name → ``True`` (delivered) or ``False``
            (send raised an exception). Disabled or filtered channels are
            not included.
        """
        results: dict[str, bool] = {}
        notif_rank = _SEVERITY_ORDER.index(notification.severity)

        for name, channel in self._channels.items():
            if not channel.enabled:
                continue
            min_rank = _SEVERITY_ORDER.index(channel.min_severity)
            if notif_rank < min_rank:
                continue
            try:
                channel.send(notification)
                results[name] = True
            except Exception as exc:
                logger.error("Channel '%s' failed: %s", name, exc)
                results[name] = False

        return results

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Toggle a channel on or off.

        Args:
            name: Channel name to update.
            enabled: ``True`` to enable, ``False`` to disable. No-ops when the
                channel is not registered.
        """
        if name in self._channels:
            self._channels[name].enabled = enabled
