"""Tests for app.health_check."""

from app.health_check import CheckResult, HealthRegistry, check


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


class TestCheckResultDetails:
    def test_check_result_defaults(self):
        r = CheckResult(name="test", healthy=True)
        assert r.message == ""
        assert r.details == {}

    def test_check_result_with_details(self):
        r = CheckResult(name="db", healthy=False, message="timeout", details={"latency_ms": 5000})
        assert r.details["latency_ms"] == 5000

    def test_health_status_failed_property(self):
        results = [make_ok("a"), make_fail("b"), make_ok("c")]
        from app.health_check import HealthStatus

        status = HealthStatus(healthy=False, results=results)
        failed = status.failed
        assert len(failed) == 1
        assert failed[0].name == "b"

    def test_multiple_failures_collected(self):
        reg = HealthRegistry()
        reg.register("x", lambda: make_fail("x"))
        reg.register("y", lambda: make_fail("y"))
        reg.register("z", lambda: make_ok("z"))
        status = reg.run()
        assert status.healthy is False
        assert len(status.failed) == 2

    def test_check_details_preserved_in_registry_run(self):
        reg = HealthRegistry()

        def detailed_check():
            return CheckResult(name="db", healthy=True, details={"pool_size": 10, "idle": 5})

        reg.register("db", detailed_check)
        status = reg.run()
        assert status.results[0].details["pool_size"] == 10

    def test_exception_message_includes_traceback(self):
        reg = HealthRegistry()

        def raises():
            raise ValueError("disk full")

        reg.register("disk", raises)
        status = reg.run()
        assert "ValueError" in status.failed[0].message

    def test_reregister_overwrites_previous(self):
        reg = HealthRegistry()
        reg.register("svc", lambda: make_fail("svc"))
        reg.register("svc", lambda: make_ok("svc"))
        status = reg.run()
        assert status.healthy is True

    def test_unregister_nonexistent_is_safe(self):
        reg = HealthRegistry()
        reg.unregister("not_there")  # should not raise
        assert len(reg) == 0
