#!/usr/bin/env python3
"""Report the latest CI workflow conclusion for every active repo.

Usage:
    python scripts/check_ci_status.py [--failing-only]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
from typing import Any

log = logging.getLogger(__name__)
GH_API = "https://api.github.com"


def _get(url: str, token: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def get_latest_runs(owner: str, repo: str, token: str) -> list[dict[str, Any]]:
    """Return the latest run per workflow for *repo*."""
    try:
        data = _get(f"{GH_API}/repos/{owner}/{repo}/actions/runs?per_page=30", token)
    except Exception as exc:
        log.warning("%s: could not fetch runs — %s", repo, exc)
        return []
    runs = data.get("workflow_runs", [])
    latest: dict[int, dict[str, Any]] = {}
    for run in runs:
        wid = run["workflow_id"]
        if wid not in latest:
            latest[wid] = run
    return list(latest.values())


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Check CI status across all repos")
    parser.add_argument("--failing-only", action="store_true", help="Only show failing workflows")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    token = os.environ.get("GH_PAT", "")
    if not token:
        log.error("GH_PAT is not set")
        return 1

    owner = os.environ.get("GITHUB_USERNAME", "atharvadevne123")
    repos_data = _get(f"{GH_API}/users/{owner}/repos?per_page=100&type=owner", token)
    repos = [r for r in repos_data if not r.get("archived") and not r.get("fork")]

    failing = 0
    for repo in sorted(repos, key=lambda r: r["name"]):
        name = repo["name"]
        runs = get_latest_runs(owner, name, token)
        if not runs:
            if not args.failing_only:
                print(f"  {name}: no CI")
            continue
        for run in runs:
            conclusion = run.get("conclusion") or run.get("status", "unknown")
            wf_name = run.get("name", "?")
            is_fail = conclusion in ("failure", "timed_out")
            if is_fail:
                failing += 1
            if not args.failing_only or is_fail:
                icon = "❌" if is_fail else ("✅" if conclusion == "success" else "⏳")
                print(f"  {icon} {name} / {wf_name}: {conclusion}")

    print(f"\n{failing} failing workflow(s) found across {len(repos)} repos")
    return 1 if failing > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
