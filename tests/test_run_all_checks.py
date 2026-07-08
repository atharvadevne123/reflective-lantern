"""Tests for scripts.run_all_checks."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def test_run_check_success() -> None:
    from scripts.run_all_checks import run_check

    ok, elapsed = run_check("echo test", [sys.executable, "-c", "print('ok')"])
    assert ok is True
    assert elapsed >= 0.0


def test_run_check_failure() -> None:
    from scripts.run_all_checks import run_check

    ok, elapsed = run_check("failing cmd", [sys.executable, "-c", "raise SystemExit(1)"])
    assert ok is False
    assert elapsed >= 0.0


def test_main_all_pass() -> None:
    from scripts import run_all_checks as rac

    passing_checks = [
        ("check1", [sys.executable, "-c", "pass"]),
        ("check2", [sys.executable, "-c", "pass"]),
    ]
    with patch.object(rac, "CHECKS", passing_checks):
        with patch.object(sys, "argv", ["run_all_checks.py"]):
            result = rac.main()
    assert result == 0


def test_main_one_fails() -> None:
    from scripts import run_all_checks as rac

    mixed_checks = [
        ("pass", [sys.executable, "-c", "pass"]),
        ("fail", [sys.executable, "-c", "raise SystemExit(1)"]),
    ]
    with patch.object(rac, "CHECKS", mixed_checks):
        with patch.object(sys, "argv", ["run_all_checks.py"]):
            result = rac.main()
    assert result == 1


def test_main_stop_on_failure() -> None:
    from scripts import run_all_checks as rac

    calls: list[str] = []

    def mock_run_check(name: str, cmd: list) -> tuple[bool, float]:
        calls.append(name)
        return name == "ok", 0.0

    checks = [("ok", []), ("bad", []), ("never", [])]
    with patch.object(rac, "CHECKS", checks):
        with patch.object(rac, "run_check", mock_run_check):
            with patch.object(sys, "argv", ["run_all_checks.py", "--stop-on-failure"]):
                result = rac.main()
    assert result == 1
    assert "never" not in calls


@pytest.mark.parametrize(
    "exit_code,expected_ok",
    [
        (0, True),
        (1, False),
        (2, False),
    ],
)
def test_run_check_exit_codes(exit_code: int, expected_ok: bool) -> None:
    from scripts.run_all_checks import run_check

    ok, elapsed = run_check(
        "parametrized", [sys.executable, "-c", f"raise SystemExit({exit_code})"]
    )
    assert ok is expected_ok
    assert elapsed >= 0.0


def test_run_check_returns_elapsed_float() -> None:
    from scripts.run_all_checks import run_check

    _, elapsed = run_check("timing check", [sys.executable, "-c", "pass"])
    assert isinstance(elapsed, float)


def test_main_timing_summary_in_output(capsys: pytest.CaptureFixture) -> None:
    from scripts import run_all_checks as rac

    checks = [("ok", [sys.executable, "-c", "pass"])]
    with patch.object(rac, "CHECKS", checks):
        with patch.object(sys, "argv", ["run_all_checks.py"]):
            rac.main()
    out = capsys.readouterr().out
    assert "total" in out.lower() or "s]" in out
