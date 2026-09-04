"""Tests for app.notification_dispatcher."""

import pytest

from app.notification_dispatcher import (
    Channel,
    Notification,
    NotificationDispatcher,
    Severity,
)


def make_channel(name="email", min_severity=Severity.INFO, enabled=True):
    received = []

    def send(n):
        received.append(n)

    ch = Channel(name=name, send=send, min_severity=min_severity, enabled=enabled)
    return ch, received


class TestDispatch:
    def test_delivers_to_matching_channel(self):
        ch, received = make_channel()
        d = NotificationDispatcher()
        d.register(ch)
        n = Notification(title="Hey", body="World")
        results = d.dispatch(n)
        assert results["email"] is True
        assert len(received) == 1

    def test_skips_below_min_severity(self):
        ch, received = make_channel(min_severity=Severity.ERROR)
        d = NotificationDispatcher()
        d.register(ch)
        n = Notification(title="Low", body="", severity=Severity.INFO)
        results = d.dispatch(n)
        assert "email" not in results
        assert len(received) == 0

    def test_delivers_above_min_severity(self):
        ch, _received = make_channel(min_severity=Severity.WARNING)
        d = NotificationDispatcher()
        d.register(ch)
        n = Notification(title="Crit", body="", severity=Severity.CRITICAL)
        results = d.dispatch(n)
        assert results["email"] is True

    def test_disabled_channel_skipped(self):
        ch, _received = make_channel(enabled=False)
        d = NotificationDispatcher()
        d.register(ch)
        n = Notification(title="x", body="")
        results = d.dispatch(n)
        assert "email" not in results

    def test_set_enabled_toggles(self):
        ch, received = make_channel(enabled=False)
        d = NotificationDispatcher()
        d.register(ch)
        d.set_enabled("email", True)
        d.dispatch(Notification(title="t", body=""))
        assert len(received) == 1

    def test_channel_exception_captured(self):
        def bad_send(n):
            raise RuntimeError("no network")

        ch = Channel(name="slack", send=bad_send)
        d = NotificationDispatcher()
        d.register(ch)
        results = d.dispatch(Notification(title="t", body=""))
        assert results["slack"] is False

    def test_unregister(self):
        ch, _received = make_channel()
        d = NotificationDispatcher()
        d.register(ch)
        d.unregister("email")
        results = d.dispatch(Notification(title="t", body=""))
        assert "email" not in results

    def test_multiple_channels(self):
        ch1, r1 = make_channel("a")
        ch2, r2 = make_channel("b")
        d = NotificationDispatcher()
        d.register(ch1)
        d.register(ch2)
        d.dispatch(Notification(title="t", body=""))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_dispatch_returns_false_on_failure(self):
        def explode(n):
            raise ValueError("boom")

        ch = Channel(name="pager", send=explode)
        d = NotificationDispatcher()
        d.register(ch)
        results = d.dispatch(Notification(title="x", body="y"))
        assert results["pager"] is False

    def test_notification_default_severity_is_info(self):
        n = Notification(title="t", body="b")
        assert n.severity == Severity.INFO

    def test_notification_body_preserved(self):
        ch, received = make_channel()
        d = NotificationDispatcher()
        d.register(ch)
        d.dispatch(Notification(title="Alert", body="Memory at 95%"))
        assert received[0].body == "Memory at 95%"

    def test_empty_dispatcher_dispatch_returns_empty(self):
        d = NotificationDispatcher()
        results = d.dispatch(Notification(title="t", body=""))
        assert results == {}

    def test_reregister_overwrites_channel(self):
        _, r1 = make_channel("x")
        ch2, r2 = make_channel("x")
        d = NotificationDispatcher()
        _, _ = make_channel("x")
        d.register(Channel(name="x", send=lambda n: r1.append(n)))
        d.register(ch2)
        d.dispatch(Notification(title="t", body=""))
        assert len(r2) == 1

    @pytest.mark.parametrize("severity", [Severity.INFO, Severity.WARNING, Severity.ERROR, Severity.CRITICAL])
    def test_all_severities_accepted(self, severity):
        ch, received = make_channel(min_severity=Severity.INFO)
        d = NotificationDispatcher()
        d.register(ch)
        d.dispatch(Notification(title="t", body="", severity=severity))
        assert len(received) == 1
