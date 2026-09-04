"""Tests for app.event_bus module."""

from __future__ import annotations

import pytest

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


class TestHandlerIsolation:
    """A failing handler must not stop the others."""

    def test_failing_handler_does_not_block_later_handlers(self) -> None:
        bus = EventBus()
        seen: list[str] = []

        def explodes(event, payload) -> None:
            raise RuntimeError("handler bug")

        bus.subscribe("tick", explodes)
        bus.subscribe("tick", lambda e, p: seen.append("second"))
        bus.publish("tick")
        assert seen == ["second"]

    def test_failing_handler_is_not_counted_as_called(self) -> None:
        bus = EventBus()

        def explodes(event, payload) -> None:
            raise RuntimeError("handler bug")

        bus.subscribe("tick", explodes)
        bus.subscribe("tick", lambda e, p: None)
        assert bus.publish("tick") == 1

    def test_failure_is_logged(self, caplog) -> None:
        bus = EventBus()

        def explodes(event, payload) -> None:
            raise RuntimeError("handler bug")

        bus.subscribe("tick", explodes)
        bus.publish("tick")
        assert "handler bug" in caplog.text

    def test_publish_survives_all_handlers_failing(self) -> None:
        bus = EventBus()

        def explodes(event, payload) -> None:
            raise RuntimeError("down")

        bus.subscribe("tick", explodes)
        bus.subscribe("tick", explodes)
        assert bus.publish("tick") == 0


class TestWildcardHandlers:
    def test_wildcard_receives_every_event(self) -> None:
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe("*", lambda e, p: seen.append(e))
        bus.publish("alpha")
        bus.publish("beta")
        assert seen == ["alpha", "beta"]

    def test_wildcard_runs_alongside_specific_handler(self) -> None:
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe("tick", lambda e, p: seen.append("specific"))
        bus.subscribe("*", lambda e, p: seen.append("wildcard"))
        assert bus.publish("tick") == 2
        assert set(seen) == {"specific", "wildcard"}

    def test_wildcard_excluded_from_listener_count(self) -> None:
        bus = EventBus()
        bus.subscribe("*", lambda e, p: None)
        assert bus.listener_count("tick") == 0

    def test_unsubscribing_wildcard_stops_delivery(self) -> None:
        bus = EventBus()
        seen: list[str] = []

        def handler(event, payload) -> None:
            seen.append(event)

        bus.subscribe("*", handler)
        assert bus.unsubscribe("*", handler) is True
        bus.publish("tick")
        assert seen == []


class TestUnsubscribeEdgeCases:
    def test_returns_false_for_unknown_handler(self) -> None:
        bus = EventBus()
        assert bus.unsubscribe("tick", lambda e, p: None) is False

    def test_returns_false_for_unknown_wildcard_handler(self) -> None:
        bus = EventBus()
        assert bus.unsubscribe("*", lambda e, p: None) is False

    def test_removes_only_the_named_handler(self) -> None:
        bus = EventBus()
        seen: list[str] = []

        def first(event, payload) -> None:
            seen.append("first")

        def second(event, payload) -> None:
            seen.append("second")

        bus.subscribe("tick", first)
        bus.subscribe("tick", second)
        bus.unsubscribe("tick", first)
        bus.publish("tick")
        assert seen == ["second"]

    def test_same_handler_subscribed_twice_needs_two_removals(self) -> None:
        bus = EventBus()

        def handler(event, payload) -> None:
            pass

        bus.subscribe("tick", handler)
        bus.subscribe("tick", handler)
        assert bus.listener_count("tick") == 2
        bus.unsubscribe("tick", handler)
        assert bus.listener_count("tick") == 1


class TestClearBehaviour:
    def test_clearing_one_event_leaves_others(self) -> None:
        bus = EventBus()
        bus.subscribe("alpha", lambda e, p: None)
        bus.subscribe("beta", lambda e, p: None)
        bus.clear("alpha")
        assert bus.listener_count("alpha") == 0
        assert bus.listener_count("beta") == 1

    def test_clearing_everything_removes_wildcards_too(self) -> None:
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe("tick", lambda e, p: seen.append("specific"))
        bus.subscribe("*", lambda e, p: seen.append("wildcard"))
        bus.clear()
        assert bus.publish("tick") == 0
        assert seen == []

    def test_clearing_wildcards_leaves_specific_handlers(self) -> None:
        bus = EventBus()
        bus.subscribe("tick", lambda e, p: None)
        bus.subscribe("*", lambda e, p: None)
        bus.clear("*")
        assert bus.publish("tick") == 1

    def test_clearing_unknown_event_is_harmless(self) -> None:
        bus = EventBus()
        bus.clear("never-registered")
        assert bus.listener_count("never-registered") == 0


class TestPublishSemantics:
    def test_publish_with_no_handlers_returns_zero(self) -> None:
        assert EventBus().publish("nobody-listening") == 0

    def test_payload_is_passed_through_unchanged(self) -> None:
        bus = EventBus()
        received: list[object] = []
        payload = {"building_id": "bldg-001", "kwh": 42.0}
        bus.subscribe("reading", lambda e, p: received.append(p))
        bus.publish("reading", payload)
        assert received == [payload]

    def test_default_payload_is_none(self) -> None:
        bus = EventBus()
        received: list[object] = []
        bus.subscribe("tick", lambda e, p: received.append(p))
        bus.publish("tick")
        assert received == [None]

    def test_handlers_run_in_registration_order(self) -> None:
        bus = EventBus()
        order: list[int] = []
        for index in range(5):
            bus.subscribe("tick", lambda e, p, i=index: order.append(i))
        bus.publish("tick")
        assert order == [0, 1, 2, 3, 4]

    def test_subscribing_during_publish_does_not_affect_current_dispatch(self) -> None:
        # publish() snapshots its handler list, so a late subscriber waits.
        bus = EventBus()
        seen: list[str] = []

        def adds_another(event, payload) -> None:
            seen.append("first")
            bus.subscribe("tick", lambda e, p: seen.append("late"))

        bus.subscribe("tick", adds_another)
        bus.publish("tick")
        assert seen == ["first"]
        bus.publish("tick")
        assert seen == ["first", "first", "late"]


class TestDefaultBus:
    def test_get_bus_returns_the_same_instance(self) -> None:
        assert get_bus() is get_bus()

    def test_default_bus_is_usable(self) -> None:
        bus = get_bus()
        bus.clear()
        seen: list[str] = []
        bus.subscribe("tick", lambda e, p: seen.append(e))
        try:
            bus.publish("tick")
            assert seen == ["tick"]
        finally:
            bus.clear()


class TestEventBusClearAndCount:
    def test_clear_specific_event(self) -> None:
        bus = EventBus()
        bus.subscribe("a", lambda e, p: None)
        bus.subscribe("b", lambda e, p: None)
        bus.clear("a")
        assert bus.listener_count("a") == 0
        assert bus.listener_count("b") == 1

    def test_clear_all_events(self) -> None:
        bus = EventBus()
        bus.subscribe("a", lambda e, p: None)
        bus.subscribe("b", lambda e, p: None)
        bus.clear()
        assert bus.listener_count("a") == 0
        assert bus.listener_count("b") == 0

    def test_listener_count_after_unsubscribe(self) -> None:
        bus = EventBus()
        h = lambda e, p: None  # noqa: E731
        bus.subscribe("evt", h)
        assert bus.listener_count("evt") == 1
        bus.unsubscribe("evt", h)
        assert bus.listener_count("evt") == 0

    def test_multiple_subscribers_counted(self) -> None:
        bus = EventBus()
        for _ in range(5):
            bus.subscribe("evt", lambda e, p: None)
        assert bus.listener_count("evt") == 5

    @pytest.mark.parametrize("event_name", ["a", "user.login", "data:received", "x.y.z"])
    def test_zero_count_for_unregistered_events(self, event_name: str) -> None:
        bus = EventBus()
        assert bus.listener_count(event_name) == 0

    def test_clear_wildcard_handlers(self) -> None:
        bus = EventBus()
        received: list[str] = []
        bus.subscribe("*", lambda e, p: received.append(e))
        bus.clear("*")
        bus.publish("any")
        assert received == []
