"""Tests for app.notification_dispatcher."""

import pytest

from app.notification_dispatcher import (
    Channel,
    Notification,
    NotificationDispatcher,
    Severity,
)


def make_channel(name="email", min_severity=Severity.INFO, enabled=True) -> None:
    received = []

    def send(n) -> None:
        received.append(n)

    ch = Channel(name=name, send=send, min_severity=min_severity, enabled=enabled)
    return ch, received


class TestDispatch:
    def test_delivers_to_matching_channel(self) -> None:
        ch, received = make_channel()
        d = NotificationDispatcher()
        d.register(ch)
        n = Notification(title="Hey", body="World")
        results = d.dispatch(n)
        assert results["email"] is True
        assert len(received) == 1

    def test_skips_below_min_severity(self) -> None:
        ch, received = make_channel(min_severity=Severity.ERROR)
        d = NotificationDispatcher()
        d.register(ch)
        n = Notification(title="Low", body="", severity=Severity.INFO)
        results = d.dispatch(n)
        assert "email" not in results
        assert len(received) == 0

    def test_delivers_above_min_severity(self) -> None:
        ch, _received = make_channel(min_severity=Severity.WARNING)
        d = NotificationDispatcher()
        d.register(ch)
        n = Notification(title="Crit", body="", severity=Severity.CRITICAL)
        results = d.dispatch(n)
        assert results["email"] is True

    def test_disabled_channel_skipped(self) -> None:
        ch, _received = make_channel(enabled=False)
        d = NotificationDispatcher()
        d.register(ch)
        n = Notification(title="x", body="")
        results = d.dispatch(n)
        assert "email" not in results

    def test_set_enabled_toggles(self) -> None:
        ch, received = make_channel(enabled=False)
        d = NotificationDispatcher()
        d.register(ch)
        d.set_enabled("email", True)
        d.dispatch(Notification(title="t", body=""))
        assert len(received) == 1

    def test_channel_exception_captured(self) -> None:
        def bad_send(n) -> None:
            raise RuntimeError("no network")

        ch = Channel(name="slack", send=bad_send)
        d = NotificationDispatcher()
        d.register(ch)
        results = d.dispatch(Notification(title="t", body=""))
        assert results["slack"] is False

    def test_unregister(self) -> None:
        ch, _received = make_channel()
        d = NotificationDispatcher()
        d.register(ch)
        d.unregister("email")
        results = d.dispatch(Notification(title="t", body=""))
        assert "email" not in results

    def test_multiple_channels(self) -> None:
        ch1, r1 = make_channel("a")
        ch2, r2 = make_channel("b")
        d = NotificationDispatcher()
        d.register(ch1)
        d.register(ch2)
        d.dispatch(Notification(title="t", body=""))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_dispatch_returns_false_on_failure(self) -> None:
        def explode(n) -> None:
            raise ValueError("boom")

        ch = Channel(name="pager", send=explode)
        d = NotificationDispatcher()
        d.register(ch)
        results = d.dispatch(Notification(title="x", body="y"))
        assert results["pager"] is False

    def test_notification_default_severity_is_info(self) -> None:
        n = Notification(title="t", body="b")
        assert n.severity == Severity.INFO

    def test_notification_body_preserved(self) -> None:
        ch, received = make_channel()
        d = NotificationDispatcher()
        d.register(ch)
        d.dispatch(Notification(title="Alert", body="Memory at 95%"))
        assert received[0].body == "Memory at 95%"

    def test_empty_dispatcher_dispatch_returns_empty(self) -> None:
        d = NotificationDispatcher()
        results = d.dispatch(Notification(title="t", body=""))
        assert results == {}

    def test_reregister_overwrites_channel(self) -> None:
        _, r1 = make_channel("x")
        ch2, r2 = make_channel("x")
        d = NotificationDispatcher()
        _, _ = make_channel("x")
        d.register(Channel(name="x", send=lambda n: r1.append(n)))
        d.register(ch2)
        d.dispatch(Notification(title="t", body=""))
        assert len(r2) == 1

    @pytest.mark.parametrize(
        "severity", [Severity.INFO, Severity.WARNING, Severity.ERROR, Severity.CRITICAL]
    )
    def test_all_severities_accepted(self, severity) -> None:
        ch, received = make_channel(min_severity=Severity.INFO)
        d = NotificationDispatcher()
        d.register(ch)
        d.dispatch(Notification(title="t", body="", severity=severity))
        assert len(received) == 1


@pytest.mark.parametrize("n_channels", [1, 3, 5])
def test_dispatch_reaches_all_channels(n_channels: int) -> None:
    """dispatch delivers to every registered channel."""
    results = {}
    dispatcher = NotificationDispatcher()
    for i in range(n_channels):
        ch, received = make_channel(name=f"ch_{i}")
        dispatcher.register(ch)
        results[f"ch_{i}"] = received
    dispatcher.dispatch(Notification(title="Test", body="hello"))
    assert all(len(v) == 1 for v in results.values())


@pytest.mark.parametrize("title", ["Alert", "Warning", "Info"])
def test_notification_title_preserved_in_channel(title: str) -> None:
    """The notification title is preserved when delivered to a channel."""
    ch, received = make_channel()
    d = NotificationDispatcher()
    d.register(ch)
    d.dispatch(Notification(title=title, body="body"))
    assert received[0].title == title


class TestDispatcherUnregister:
    def test_unregistered_channel_receives_nothing(self) -> None:
        ch, received = make_channel()
        d = NotificationDispatcher()
        d.register(ch)
        d.unregister("email")
        d.dispatch(Notification(title="t", body=""))
        assert received == []


class TestDispatcherChannelNames:
    def test_empty_dispatcher_has_no_names(self) -> None:
        d = NotificationDispatcher()
        assert d.channel_names() == []

    def test_returns_registered_names(self) -> None:
        d = NotificationDispatcher()
        ch1, _ = make_channel("alpha")
        ch2, _ = make_channel("beta")
        d.register(ch1)
        d.register(ch2)
        names = d.channel_names()
        assert "alpha" in names
        assert "beta" in names

    def test_unregister_removes_name(self) -> None:
        d = NotificationDispatcher()
        ch, _ = make_channel("to_remove")
        d.register(ch)
        d.unregister("to_remove")
        assert "to_remove" not in d.channel_names()


class TestDispatcherEnabledChannels:
    def test_disabled_channel_not_in_enabled(self) -> None:
        d = NotificationDispatcher()
        ch, _ = make_channel("off", enabled=False)
        d.register(ch)
        assert "off" not in d.enabled_channels()

    def test_enabled_channel_in_enabled(self) -> None:
        d = NotificationDispatcher()
        ch, _ = make_channel("on", enabled=True)
        d.register(ch)
        assert "on" in d.enabled_channels()

    def test_toggle_enabled_updates_list(self) -> None:
        d = NotificationDispatcher()
        ch, _ = make_channel("toggle", enabled=False)
        d.register(ch)
        assert "toggle" not in d.enabled_channels()
        d.set_enabled("toggle", True)
        assert "toggle" in d.enabled_channels()


class TestDispatcherLen:
    def test_empty_len_zero(self) -> None:
        d = NotificationDispatcher()
        assert len(d) == 0

    def test_len_increments_on_register(self) -> None:
        d = NotificationDispatcher()
        ch, _ = make_channel("ch1")
        d.register(ch)
        assert len(d) == 1

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_len_matches_registered_count(self, n: int) -> None:
        d = NotificationDispatcher()
        for i in range(n):
            ch, _ = make_channel(f"ch{i}")
            d.register(ch)
        assert len(d) == n


class TestDisabledChannels:
    def test_no_channels_returns_empty(self) -> None:
        d = NotificationDispatcher()
        assert d.disabled_channels() == []

    def test_all_enabled_returns_empty(self) -> None:
        d = NotificationDispatcher()
        ch, _ = make_channel("email", enabled=True)
        d.register(ch)
        assert d.disabled_channels() == []

    def test_disabled_channel_appears(self) -> None:
        d = NotificationDispatcher()
        ch, _ = make_channel("slack", enabled=False)
        d.register(ch)
        assert "slack" in d.disabled_channels()

    def test_mix_of_enabled_and_disabled(self) -> None:
        d = NotificationDispatcher()
        ch_on, _ = make_channel("email", enabled=True)
        ch_off, _ = make_channel("sms", enabled=False)
        d.register(ch_on)
        d.register(ch_off)
        assert d.disabled_channels() == ["sms"]
        assert "email" not in d.disabled_channels()

    @pytest.mark.parametrize("n_disabled", [0, 1, 3])
    def test_count_matches_disabled_n(self, n_disabled: int) -> None:
        d = NotificationDispatcher()
        for i in range(n_disabled):
            ch, _ = make_channel(f"ch{i}", enabled=False)
            d.register(ch)
        assert len(d.disabled_channels()) == n_disabled
