"""Multi-channel notification dispatcher.

Routes notifications to one or more registered channels (e.g. email,
Slack, webhook) based on severity and channel availability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List

__all__ = [
    "Severity",
    "Notification",
    "Channel",
    "NotificationDispatcher",
]

logger = logging.getLogger(__name__)


class Severity(str, Enum):
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
    tags: List[str] = field(default_factory=list)


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
        self._channels: Dict[str, Channel] = {}

    def register(self, channel: Channel) -> None:
        """Add or replace a channel by name."""
        self._channels[channel.name] = channel

    def unregister(self, name: str) -> None:
        """Remove a channel by name."""
        self._channels.pop(name, None)

    def dispatch(self, notification: Notification) -> Dict[str, bool]:
        """Send *notification* to all eligible channels.

        Returns:
            Mapping of channel name to delivery success.
        """
        results: Dict[str, bool] = {}
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
        """Toggle a channel on or off."""
        if name in self._channels:
            self._channels[name].enabled = enabled
