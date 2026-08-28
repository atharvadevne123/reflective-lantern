"""Tests for app.notification_dispatcher."""


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
        ch, received = make_channel(min_severity=Severity.WARNING)
        d = NotificationDispatcher()
        d.register(ch)
        n = Notification(title="Crit", body="", severity=Severity.CRITICAL)
        results = d.dispatch(n)
        assert results["email"] is True

    def test_disabled_channel_skipped(self):
        ch, received = make_channel(enabled=False)
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
        ch, received = make_channel()
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
