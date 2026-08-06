"""Container health check script for Forge-Guard.

Intended to be called by Docker's HEALTHCHECK instruction:

    HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
        CMD python scripts/healthcheck.py

Exits 0 on healthy, 1 on degraded/unreachable.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request


def check(host: str = "localhost", port: int = 8000, timeout: int = 5) -> int:
    """Probe the /health endpoint and return an exit code.

    Args:
        host: Service hostname.
        port: Service port.
        timeout: Request timeout in seconds.

    Returns:
        0 if the response is 200 and status is "healthy", 1 otherwise.
    """
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                print(f"HEALTH CHECK FAILED: HTTP {resp.status}", file=sys.stderr)
                return 1
            import json

            body = json.loads(resp.read())
            if body.get("status") == "healthy":
                print(f"OK: {body}")
                return 0
            print(f"DEGRADED: {body}", file=sys.stderr)
            return 1
    except urllib.error.URLError as exc:
        print(f"HEALTH CHECK UNREACHABLE: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"HEALTH CHECK ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import os

    host = os.getenv("APP_HOST", "localhost")
    port = int(os.getenv("APP_PORT", "8000"))
    sys.exit(check(host=host, port=port))
