"""Simple health check script for the Energy-Seer API."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request


def check(url: str = "http://localhost:8000/api/v1/health") -> int:
    """Ping the health endpoint and return an exit code (0=ok, 1=fail)."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print(f"OK: {url} responded {resp.status}")
                return 0
            print(f"FAIL: {url} responded {resp.status}")
            return 1
    except urllib.error.URLError as exc:
        print(f"FAIL: {url} unreachable — {exc}")
        return 1


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/api/v1/health"
    sys.exit(check(target))
