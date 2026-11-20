"""Run a suite of pre-flight checks and report results."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Any

CHECKS: list[tuple[str, list[Any]]] = [
    ("ruff", ["ruff", "check", "."]),
    ("pytest", [sys.executable, "-m", "pytest", "--tb=short", "-q"]),
]


def run_check(name: str, cmd: list) -> tuple[bool, float]:
    """Run *cmd* and return (success, elapsed_seconds)."""
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True)
        ok = result.returncode == 0
    except Exception:
        ok = False
    elapsed = time.monotonic() - t0
    return ok, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all pre-flight checks.")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    results: list[tuple[str, bool, float]] = []
    for name, cmd in CHECKS:
        ok, elapsed = run_check(name, cmd)
        results.append((name, ok, elapsed))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}  ({elapsed:.2f}s)")
        if not ok and args.stop_on_failure:
            break

    total = sum(e for _, _, e in results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    print(f"\nTotal: {passed} passed, {failed} failed  [{total:.2f}s]")

    return 0 if all(ok for _, ok, _ in results) else 1


__all__ = ["CHECKS", "run_check"]


if __name__ == "__main__":
    sys.exit(main())
