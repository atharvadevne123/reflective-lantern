#!/usr/bin/env python3
"""Run all local health checks in sequence.

Usage:
    python scripts/run_all_checks.py [--stop-on-failure]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from config.constants import SEPARATOR_WIDTH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SCRIPTS_DIR = Path(__file__).parent
CHECKS = [
    ("History validation", [sys.executable, str(SCRIPTS_DIR / "validate_history.py")]),
    ("History summary", [sys.executable, str(SCRIPTS_DIR / "summarize_history.py")]),
    (
        "Foundry export",
        [sys.executable, str(SCRIPTS_DIR / "foundry_sync.py"), "--export-only"],
    ),
]


def run_check(name: str, cmd: list[str]) -> tuple[bool, float]:
    """Run a single check. Return (passed, elapsed_seconds)."""
    logger.info("─" * SEPARATOR_WIDTH)
    logger.info("  %s", name)
    logger.info("─" * SEPARATOR_WIDTH)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode == 0:
        logger.info("  ✓ %s passed (%.2fs)", name, elapsed)
    else:
        logger.error("  ✗ %s failed (exit %d, %.2fs)", name, result.returncode, elapsed)
    return result.returncode == 0, elapsed


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
    total_time = 0.0
    for name, cmd in CHECKS:
        ok, elapsed = run_check(name, cmd)
        total_time += elapsed
        if ok:
            passed += 1
        else:
            failed += 1
            if args.stop_on_failure:
                break

    print(f"\n{'═' * SEPARATOR_WIDTH}")
    print(f"  {passed} check(s) passed, {failed} failed  [{total_time:.2f}s total]")
    print(f"{'═' * SEPARATOR_WIDTH}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
