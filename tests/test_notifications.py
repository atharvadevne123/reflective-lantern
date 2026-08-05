"""Tests for app/notifications.py alert and notification utilities."""

from __future__ import annotations

import pytest

from app.notifications import (
    Alert,
    AlertQueue,
    make_anomaly_alert,
    make_drift_alert,
    severity_rank,
)


def test_alert_to_dict_has_required_keys():
    alert = Alert(severity="warning", message="test")
    d = alert.to_dict()
    assert "severity" in d
    assert "message" in d
    assert "source" in d
    assert "tags" in d
    assert "metadata" in d


def test_severity_rank_ordering():
    assert severity_rank("info") < severity_rank("warning") < severity_rank("critical")


def test_severity_rank_unknown_returns_negative():
    assert severity_rank("unknown") < 0


@pytest.mark.parametrize("sev", ["info", "warning", "critical"])
def test_severity_rank_valid_severities(sev):
    assert severity_rank(sev) >= 0


def test_alert_queue_push_and_len():
    q = AlertQueue()
    q.push(Alert(severity="info", message="m1"))
    q.push(Alert(severity="warning", message="m2"))
    assert len(q) == 2


def test_alert_queue_max_size_evicts_oldest():
    q = AlertQueue(max_size=3)
    for i in range(5):
        q.push(Alert(severity="info", message=f"msg-{i}"))
    assert len(q) == 3
    assert q.alerts[0].message == "msg-2"


def test_alert_queue_filter_by_severity():
    q = AlertQueue()
    q.push(Alert(severity="info", message="i"))
    q.push(Alert(severity="warning", message="w"))
    q.push(Alert(severity="critical", message="c"))
    result = q.filter_by_severity("warning")
    assert all(severity_rank(a.severity) >= severity_rank("warning") for a in result)
    assert len(result) == 2


def test_alert_queue_filter_by_tag():
    q = AlertQueue()
    q.push(Alert(severity="info", message="a", tags=["anomaly"]))
    q.push(Alert(severity="info", message="b", tags=["drift"]))
    result = q.filter_by_tag("anomaly")
    assert len(result) == 1
    assert result[0].message == "a"


def test_alert_queue_summary_counts():
    q = AlertQueue()
    q.push(Alert(severity="info", message="i"))
    q.push(Alert(severity="critical", message="c1"))
    q.push(Alert(severity="critical", message="c2"))
    s = q.summary()
    assert s["info"] == 1
    assert s["critical"] == 2
    assert s["warning"] == 0


def test_alert_queue_clear():
    q = AlertQueue()
    q.push(Alert(severity="info", message="x"))
    q.clear()
    assert len(q) == 0


def test_make_anomaly_alert_warning():
    a = make_anomaly_alert("bldg-1", 0.75, is_critical=False)
    assert a.severity == "warning"
    assert "bldg-1" in a.message
    assert a.metadata["anomaly_score"] == pytest.approx(0.75)


def test_make_anomaly_alert_critical():
    a = make_anomaly_alert("bldg-2", 0.95, is_critical=True)
    assert a.severity == "critical"


def test_make_drift_alert():
    a = make_drift_alert(0.42, 0.03)
    assert a.severity == "warning"
    assert "drift" in a.tags
    assert a.metadata["ks_statistic"] == pytest.approx(0.42)


@pytest.mark.parametrize("n_alerts", [1, 5, 10])
def test_alert_queue_length_parametrize(n_alerts):
    q = AlertQueue(max_size=100)
    for i in range(n_alerts):
        q.push(Alert(severity="info", message=f"m{i}"))
    assert len(q) == n_alerts


def test_alert_to_dict_severity_matches():
    for sev in ("info", "warning", "critical"):
        a = Alert(severity=sev, message="test")
        assert a.to_dict()["severity"] == sev


def test_alert_with_custom_source():
    a = Alert(severity="info", message="msg", source="sensor-42")
    d = a.to_dict()
    assert d["source"] == "sensor-42"


def test_alert_with_metadata():
    a = Alert(severity="warning", message="msg", metadata={"score": 0.9})
    assert a.to_dict()["metadata"]["score"] == pytest.approx(0.9)


def test_alert_queue_summary_empty():
    q = AlertQueue()
    s = q.summary()
    assert s["info"] == 0
    assert s["warning"] == 0
    assert s["critical"] == 0


def test_alert_queue_filter_no_match():
    q = AlertQueue()
    q.push(Alert(severity="info", message="x"))
    result = q.filter_by_severity("critical")
    assert result == []


def test_make_drift_alert_has_ks_stat():
    a = make_drift_alert(0.7, 0.01)
    assert "ks_statistic" in a.metadata


@pytest.mark.parametrize(
    "score,is_critical,expected_sev",
    [
        (0.5, False, "warning"),
        (0.95, True, "critical"),
    ],
)
def test_make_anomaly_alert_severity(score, is_critical, expected_sev):
    a = make_anomaly_alert("zone-1", score, is_critical=is_critical)
    assert a.severity == expected_sev


class TestAlertSummary:
    def test_empty(self) -> None:
        from app.notifications import alert_summary

        result = alert_summary([])
        assert result == {"info": 0, "warning": 0, "critical": 0}

    def test_counts(self) -> None:
        from app.notifications import Alert, alert_summary

        alerts = [
            Alert("info", "i"),
            Alert("warning", "w"),
            Alert("warning", "w2"),
            Alert("critical", "c"),
        ]
        result = alert_summary(alerts)
        assert result["info"] == 1
        assert result["warning"] == 2
        assert result["critical"] == 1

    def test_unknown_severity_not_counted(self) -> None:
        from app.notifications import Alert, alert_summary

        alerts = [Alert("unknown", "u")]
        result = alert_summary(alerts)
        assert sum(result.values()) == 0


class TestHighestSeverity:
    def test_empty_returns_none(self) -> None:
        from app.notifications import highest_severity

        assert highest_severity([]) == "none"

    def test_returns_critical(self) -> None:
        from app.notifications import Alert, highest_severity

        alerts = [Alert("info", "i"), Alert("critical", "c")]
        assert highest_severity(alerts) == "critical"

    def test_returns_warning(self) -> None:
        from app.notifications import Alert, highest_severity

        alerts = [Alert("info", "i"), Alert("warning", "w")]
        assert highest_severity(alerts) == "warning"

    def test_all_info(self) -> None:
        from app.notifications import Alert, highest_severity

        alerts = [Alert("info", "a"), Alert("info", "b")]
        assert highest_severity(alerts) == "info"


class TestFilterAlertsBySeverity:
    def test_filters_info(self) -> None:
        from app.notifications import Alert, filter_alerts_by_severity

        alerts = [
            Alert("info", "msg1"),
            Alert("critical", "msg2"),
            Alert("warning", "msg3"),
        ]
        result = filter_alerts_by_severity(alerts, "critical")
        assert all(a.severity == "critical" for a in result)

    def test_empty_input(self) -> None:
        from app.notifications import filter_alerts_by_severity

        assert filter_alerts_by_severity([], "info") == []

    def test_all_pass_info_threshold(self) -> None:
        from app.notifications import Alert, filter_alerts_by_severity

        alerts = [Alert("warning", "x"), Alert("critical", "y")]
        result = filter_alerts_by_severity(alerts, "info")
        assert len(result) == 2


class TestDeduplicateAlerts:
    def test_removes_duplicates(self) -> None:
        from app.notifications import Alert, deduplicate_alerts

        alerts = [Alert("info", "dup"), Alert("critical", "dup"), Alert("info", "unique")]
        result = deduplicate_alerts(alerts)
        assert len(result) == 2

    def test_empty(self) -> None:
        from app.notifications import deduplicate_alerts

        assert deduplicate_alerts([]) == []

    def test_no_duplicates(self) -> None:
        from app.notifications import Alert, deduplicate_alerts

        alerts = [Alert("info", "a"), Alert("info", "b")]
        assert len(deduplicate_alerts(alerts)) == 2


class TestFormatAlertText:
    def test_format_contains_severity(self) -> None:
        from app.notifications import Alert, format_alert_text

        a = Alert("warning", "Drift detected")
        result = format_alert_text(a)
        assert "WARNING" in result
        assert "Drift detected" in result

    def test_format_structure(self) -> None:
        from app.notifications import Alert, format_alert_text

        a = Alert("info", "test message")
        result = format_alert_text(a)
        assert result.startswith("[")


class TestBatchAlertsBySeverity:
    def test_groups_correctly(self) -> None:
        from app.notifications import Alert, batch_alerts_by_severity

        alerts = [Alert("info", "a"), Alert("critical", "b"), Alert("info", "c")]
        batches = batch_alerts_by_severity(alerts)
        assert len(batches["info"]) == 2
        assert len(batches["critical"]) == 1

    def test_empty(self) -> None:
        from app.notifications import batch_alerts_by_severity

        assert batch_alerts_by_severity([]) == {}

    def test_single_severity(self) -> None:
        from app.notifications import Alert, batch_alerts_by_severity

        alerts = [Alert("critical", "x"), Alert("critical", "y")]
        batches = batch_alerts_by_severity(alerts)
        assert "critical" in batches
        assert len(batches["critical"]) == 2
