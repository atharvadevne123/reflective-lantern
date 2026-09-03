"""Tests for app.health_check."""

from app.health_check import CheckResult, HealthRegistry, check
import pytest



def make_ok(name="db") -> CheckResult:
    return CheckResult(name=name, healthy=True, message="ok")


def make_fail(name="cache") -> CheckResult:
    return CheckResult(name=name, healthy=False, message="timeout")


class TestHealthRegistry:
    def test_empty_registry_is_healthy(self):
        reg = HealthRegistry()
        status = reg.run()
        assert status.healthy is True
        assert status.results == []

    def test_all_pass(self):
        reg = HealthRegistry()
        reg.register("db", lambda: make_ok("db"))
        reg.register("cache", lambda: make_ok("cache"))
        status = reg.run()
        assert status.healthy is True
        assert len(status.failed) == 0

    def test_one_fail_makes_unhealthy(self):
        reg = HealthRegistry()
        reg.register("db", lambda: make_ok("db"))
        reg.register("cache", lambda: make_fail("cache"))
        status = reg.run()
        assert status.healthy is False
        assert len(status.failed) == 1
        assert status.failed[0].name == "cache"

    def test_exception_becomes_failure(self):
        reg = HealthRegistry()

        def bad_check():
            raise RuntimeError("boom")

        reg.register("bad", bad_check)
        status = reg.run()
        assert status.healthy is False
        assert "boom" in status.failed[0].message

    def test_unregister(self):
        reg = HealthRegistry()
        reg.register("x", lambda: make_fail("x"))
        reg.unregister("x")
        assert len(reg) == 0
        status = reg.run()
        assert status.healthy is True

    def test_len(self):
        reg = HealthRegistry()
        reg.register("a", lambda: make_ok("a"))
        reg.register("b", lambda: make_ok("b"))
        assert len(reg) == 2


class TestCheckDecorator:
    def test_decorator_registers(self):
        reg = HealthRegistry()

        @check("ping", registry=reg)
        def ping_check():
            return CheckResult(name="ping", healthy=True)

        status = reg.run()
        assert status.healthy is True
        assert status.results[0].name == "ping"

    def test_decorator_returns_original_fn(self):
        reg = HealthRegistry()

        @check("noop", registry=reg)
        def noop():
            return CheckResult(name="noop", healthy=True)

        result = noop()
        assert result.name == "noop"


class TestHealthRegistryExtended:
    def test_run_empty_registry_is_healthy(self):
        from app.health_check import HealthRegistry
        reg = HealthRegistry()
        status = reg.run()
        assert status.healthy is True
        assert status.results == []

    def test_failed_property_filters_unhealthy(self):
        from app.health_check import CheckResult, HealthRegistry
        reg = HealthRegistry()
        reg.register("ok", lambda: CheckResult("ok", True, "fine"))
        reg.register("bad", lambda: CheckResult("bad", False, "broken"))
        status = reg.run()
        assert len(status.failed) == 1
        assert status.failed[0].name == "bad"

    def test_unregister_removes_check(self):
        from app.health_check import CheckResult, HealthRegistry
        reg = HealthRegistry()
        reg.register("x", lambda: CheckResult("x", True))
        assert len(reg) == 1
        reg.unregister("x")
        assert len(reg) == 0

    def test_exception_in_check_captured_as_unhealthy(self):
        from app.health_check import HealthRegistry
        reg = HealthRegistry()
        def boom() -> None:
            raise RuntimeError("crash")
        reg.register("boom", boom)
        status = reg.run()
        assert not status.healthy
        assert "RuntimeError" in status.results[0].message

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_all_healthy_checks_aggregate_healthy(self, n: int):
        from app.health_check import CheckResult, HealthRegistry
        reg = HealthRegistry()
        for i in range(n):
            name = f"check_{i}"
            reg.register(name, lambda _name=name: CheckResult(_name, True))
        status = reg.run()
        assert status.healthy is True
        assert len(status.results) == n
