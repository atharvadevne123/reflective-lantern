"""Tests for app.alerting module."""

from __future__ import annotations

import pytest

from app.alerting import Alert, AlertManager, AlertRule, Severity

BASE_NOW = 1_000_000.0


def _make_rule(**kwargs) -> AlertRule:
    defaults = {"name": "r1", "metric": "cpu", "threshold": 80.0, "cooldown_s": 60}
    defaults.update(kwargs)
    return AlertRule(**defaults)


class TestAlertRuleEvaluate:
    @pytest.mark.parametrize(
        "op,value,threshold,fires",
        [
            (">", 90, 80, True),
            (">", 80, 80, False),
            (">=", 80, 80, True),
            ("<", 70, 80, True),
            ("<", 90, 80, False),
            ("<=", 80, 80, True),
            ("==", 80, 80, True),
            ("==", 81, 80, False),
        ],
    )
    def test_comparison_operators(self, op, value, threshold, fires):
        rule = _make_rule(comparison=op, threshold=threshold)
        result = rule.evaluate(value, BASE_NOW)
        assert (result is not None) == fires

    def test_alert_fields_populated(self):
        rule = _make_rule()
        alert = rule.evaluate(90, BASE_NOW)
        assert alert.name == "r1"
        assert alert.metric == "cpu"
        assert alert.value == 90
        assert alert.threshold == 80.0
        assert isinstance(alert.severity, Severity)

    def test_cooldown_prevents_refiring(self):
        rule = _make_rule(cooldown_s=60)
        rule.evaluate(90, BASE_NOW)
        result = rule.evaluate(90, BASE_NOW + 30)  # within cooldown
        assert result is None

    def test_fires_again_after_cooldown(self):
        rule = _make_rule(cooldown_s=60)
        rule.evaluate(90, BASE_NOW)
        result = rule.evaluate(90, BASE_NOW + 61)  # past cooldown
        assert result is not None

    def test_unknown_operator_returns_none(self):
        rule = _make_rule(comparison="??")
        result = rule.evaluate(90, BASE_NOW)
        assert result is None


class TestAlertManager:
    def _manager(self):
        return AlertManager()

    def test_add_and_fire_rule(self):
        mgr = self._manager()
        mgr.add_rule(_make_rule())
        alerts = mgr.evaluate_all({"cpu": 90}, now=BASE_NOW)
        assert len(alerts) == 1

    def test_no_alert_when_below_threshold(self):
        mgr = self._manager()
        mgr.add_rule(_make_rule())
        alerts = mgr.evaluate_all({"cpu": 70}, now=BASE_NOW)
        assert alerts == []

    def test_missing_metric_skipped(self):
        mgr = self._manager()
        mgr.add_rule(_make_rule(metric="missing"))
        alerts = mgr.evaluate_all({"cpu": 90}, now=BASE_NOW)
        assert alerts == []

    def test_remove_rule(self):
        mgr = self._manager()
        mgr.add_rule(_make_rule())
        assert mgr.remove_rule("r1") is True
        alerts = mgr.evaluate_all({"cpu": 90}, now=BASE_NOW)
        assert alerts == []

    def test_remove_nonexistent_returns_false(self):
        mgr = self._manager()
        assert mgr.remove_rule("ghost") is False

    def test_history_accumulates(self):
        mgr = self._manager()
        mgr.add_rule(_make_rule(cooldown_s=0))
        mgr.evaluate_all({"cpu": 90}, now=BASE_NOW)
        mgr.evaluate_all({"cpu": 90}, now=BASE_NOW + 1)
        assert len(mgr.history) == 2

    def test_handler_called_on_alert(self):
        received = []
        mgr = AlertManager(handlers=[lambda a: received.append(a)])
        mgr.add_rule(_make_rule())
        mgr.evaluate_all({"cpu": 90}, now=BASE_NOW)
        assert len(received) == 1 and isinstance(received[0], Alert)

    def test_handler_exception_does_not_block(self):
        def bad_handler(a):
            raise RuntimeError("oops")

        mgr = AlertManager(handlers=[bad_handler])
        mgr.add_rule(_make_rule())
        alerts = mgr.evaluate_all({"cpu": 90}, now=BASE_NOW)
        assert len(alerts) == 1

    @pytest.mark.parametrize("severity", list(Severity))
    def test_severity_variants(self, severity):
        rule = _make_rule(severity=severity)
        alert = rule.evaluate(90, BASE_NOW)
        assert alert.severity == severity


class TestAlertManagerMultipleRules:
    def test_two_rules_both_fire(self):
        mgr = AlertManager()
        mgr.add_rule(_make_rule(name="r1", metric="cpu"))
        mgr.add_rule(_make_rule(name="r2", metric="mem"))
        alerts = mgr.evaluate_all({"cpu": 90, "mem": 90}, now=BASE_NOW)
        assert len(alerts) == 2

    def test_only_matching_rule_fires(self):
        mgr = AlertManager()
        mgr.add_rule(_make_rule(name="r1", metric="cpu"))
        mgr.add_rule(_make_rule(name="r2", metric="mem"))
        alerts = mgr.evaluate_all({"cpu": 90, "mem": 50}, now=BASE_NOW)
        assert len(alerts) == 1
        assert alerts[0].name == "r1"

    def test_replace_existing_rule(self):
        mgr = AlertManager()
        mgr.add_rule(_make_rule(name="r1", threshold=80.0))
        mgr.add_rule(_make_rule(name="r1", threshold=95.0))
        alerts = mgr.evaluate_all({"cpu": 90}, now=BASE_NOW)
        assert alerts == []

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_n_rules_all_fire(self, n):
        mgr = AlertManager()
        for i in range(n):
            mgr.add_rule(_make_rule(name=f"r{i}", metric=f"m{i}"))
        metrics = {f"m{i}": 90 for i in range(n)}
        assert len(mgr.evaluate_all(metrics, now=BASE_NOW)) == n

    def test_clear_history_via_new_manager(self):
        mgr = AlertManager()
        mgr.add_rule(_make_rule(cooldown_s=0))
        mgr.evaluate_all({"cpu": 90}, now=BASE_NOW)
        mgr.evaluate_all({"cpu": 90}, now=BASE_NOW + 1)
        assert len(mgr.history) == 2

    def test_add_rule_returns_none(self):
        mgr = AlertManager()
        result = mgr.add_rule(_make_rule())
        assert result is None

    def test_remove_rule_idempotent(self):
        mgr = AlertManager()
        mgr.add_rule(_make_rule())
        assert mgr.remove_rule("r1") is True
        assert mgr.remove_rule("r1") is False


class TestAlertRuleEdgeCases:
    def test_rule_name_preserved_in_alert(self) -> None:
        rule = _make_rule(name="my_rule", threshold=10.0)
        alert = rule.evaluate({"cpu": 20.0}, now=BASE_NOW)
        assert alert is not None
        assert alert.rule_name == "my_rule"

    def test_alert_has_metric_value(self) -> None:
        rule = _make_rule(threshold=5.0)
        alert = rule.evaluate({"cpu": 9.0}, now=BASE_NOW)
        assert alert is not None
        assert alert.value == 9.0

    @pytest.mark.parametrize("severity", ["info", "warning", "critical"])
    def test_severity_stored_on_alert(self, severity: str) -> None:
        rule = _make_rule(severity=severity, threshold=1.0)
        alert = rule.evaluate({"cpu": 2.0}, now=BASE_NOW)
        assert alert is not None
        assert alert.severity == Severity(severity)

    def test_manager_history_length_matches_fires(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(_make_rule(cooldown_s=0))
        for i in range(3):
            mgr.evaluate_all({"cpu": 100.0}, now=float(i * 1000))
        assert len(mgr.history) == 3


class TestAlertManagerRuleManagement:
    def test_remove_rule_returns_true(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(_make_rule(name="r1"))
        assert mgr.remove_rule("r1") is True

    def test_remove_nonexistent_rule_returns_false(self) -> None:
        mgr = AlertManager()
        assert mgr.remove_rule("nonexistent") is False

    def test_rule_names_sorted(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(_make_rule(name="cpu_alert"))
        mgr.add_rule(_make_rule(name="mem_alert"))
        names = mgr.rule_names()
        assert names == sorted(names)

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_rule_count_after_adds(self, n: int) -> None:
        mgr = AlertManager()
        for i in range(n):
            mgr.add_rule(_make_rule(name=f"rule_{i}"))
        assert len(mgr.rule_names()) == n


class TestAlertManagerHistory:
    def test_clear_history_empties_list(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(_make_rule(cooldown_s=0))
        mgr.evaluate_all({"cpu": 100.0}, now=BASE_NOW)
        assert len(mgr.history) > 0
        mgr.clear_history()
        assert len(mgr.history) == 0

    def test_history_for_metric_filters_correctly(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(_make_rule(name="cpu_rule", metric="cpu", cooldown_s=0))
        mgr.evaluate_all({"cpu": 100.0}, now=BASE_NOW)
        cpu_alerts = mgr.history_for_metric("cpu")
        assert len(cpu_alerts) == len(mgr.history)

    def test_history_for_unknown_metric_empty(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(_make_rule(cooldown_s=0))
        mgr.evaluate_all({"cpu": 100.0}, now=BASE_NOW)
        assert mgr.history_for_metric("memory") == []
