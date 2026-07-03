#!/usr/bin/env python3
"""Run all local health checks in sequence.

Usage:
    python scripts/run_all_checks.py [--stop-on-failure]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
CHECKS = [
    ("History validation", [sys.executable, str(SCRIPTS_DIR / "validate_history.py")]),
    ("History summary", [sys.executable, str(SCRIPTS_DIR / "summarize_history.py")]),
]


def run_check(name: str, cmd: list[str]) -> bool:
    """Run a single check. Return True if it passed."""
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print(f"  ✓ {name} passed")
    else:
        print(f"  ✗ {name} failed (exit {result.returncode})")
    return result.returncode == 0


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run all Reflective Lantern checks")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Abort after the first failing check",
    )
    args = parser.parse_args()

    passed = 0
    failed = 0
    for name, cmd in CHECKS:
        ok = run_check(name, cmd)
        if ok:
            passed += 1
        else:
            failed += 1
            if args.stop_on_failure:
                break

    print(f"\n{'═' * 60}")
    print(f"  {passed} check(s) passed, {failed} failed")
    print(f"{'═' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
