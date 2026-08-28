"""Tests for app.event_bus module."""

from __future__ import annotations

from app.event_bus import EventBus, get_bus


def _make_recorder():
    calls = []

    def handler(event, payload):
        calls.append((event, payload))

    return handler, calls


class TestEventBusSubscribePublish:
    def test_published_event_reaches_handler(self):
        bus = EventBus()
        handler, calls = _make_recorder()
        bus.subscribe("user.created", handler)
        bus.publish("user.created", {"id": 1})
        assert calls == [("user.created", {"id": 1})]

    def test_multiple_handlers_called_in_order(self):
        bus = EventBus()
        order = []
        bus.subscribe("evt", lambda e, p: order.append(1))
        bus.subscribe("evt", lambda e, p: order.append(2))
        bus.publish("evt")
        assert order == [1, 2]

    def test_returns_handler_count(self):
        bus = EventBus()
        bus.subscribe("x", lambda e, p: None)
        bus.subscribe("x", lambda e, p: None)
        assert bus.publish("x") == 2

    def test_no_handlers_returns_zero(self):
        bus = EventBus()
        assert bus.publish("unknown") == 0

    def test_handler_exception_does_not_block_others(self):
        bus = EventBus()
        results = []

        def bad(e, p):
            raise RuntimeError("bad")

        def good(e, p):
            results.append("good")

        bus.subscribe("x", bad)
        bus.subscribe("x", good)
        bus.publish("x")
        assert results == ["good"]


class TestWildcardHandler:
    def test_wildcard_receives_all_events(self):
        bus = EventBus()
        handler, calls = _make_recorder()
        bus.subscribe("*", handler)
        bus.publish("a")
        bus.publish("b")
        assert len(calls) == 2

    def test_wildcard_and_specific_both_called(self):
        bus = EventBus()
        specific, s_calls = _make_recorder()
        wildcard, w_calls = _make_recorder()
        bus.subscribe("login", specific)
        bus.subscribe("*", wildcard)
        bus.publish("login", "alice")
        assert s_calls == [("login", "alice")]
        assert w_calls == [("login", "alice")]


class TestUnsubscribe:
    def test_unsubscribe_returns_true_when_found(self):
        bus = EventBus()
        handler, _ = _make_recorder()
        bus.subscribe("x", handler)
        assert bus.unsubscribe("x", handler) is True

    def test_unsubscribe_returns_false_when_missing(self):
        bus = EventBus()
        handler, _ = _make_recorder()
        assert bus.unsubscribe("x", handler) is False

    def test_unsubscribed_handler_not_called(self):
        bus = EventBus()
        handler, calls = _make_recorder()
        bus.subscribe("x", handler)
        bus.unsubscribe("x", handler)
        bus.publish("x")
        assert calls == []


class TestClear:
    def test_clear_specific_event(self):
        bus = EventBus()
        handler, calls = _make_recorder()
        bus.subscribe("x", handler)
        bus.clear("x")
        bus.publish("x")
        assert calls == []

    def test_clear_all(self):
        bus = EventBus()
        h1, c1 = _make_recorder()
        h2, c2 = _make_recorder()
        bus.subscribe("a", h1)
        bus.subscribe("*", h2)
        bus.clear()
        bus.publish("a")
        assert c1 == [] and c2 == []


def test_listener_count():
    bus = EventBus()
    bus.subscribe("ev", lambda e, p: None)
    bus.subscribe("ev", lambda e, p: None)
    assert bus.listener_count("ev") == 2
    assert bus.listener_count("other") == 0


def test_get_bus_returns_singleton():
    assert get_bus() is get_bus()
