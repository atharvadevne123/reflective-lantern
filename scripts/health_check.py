#!/usr/bin/env python3
"""Check health of all non-archived, non-fork repos for the configured owner.

Reports failing CI workflows, missing releases, open branches, and
repos without a CI workflow. Exits non-zero if any issues are found.

Usage:
    python scripts/health_check.py [--json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)
GH_API = "https://api.github.com"

FAILING_CONCLUSIONS: frozenset[str] = frozenset({"failure", "timed_out"})
DEFAULT_RETRIES = 3
WORKFLOW_RUNS_PER_PAGE = 20
BRANCHES_PER_PAGE = 100
RELEASES_PER_PAGE = 1


@dataclass
class RepoHealth:
    """Health summary for a single repository."""

    name: str
    failing_workflows: list[str] = field(default_factory=list)
    open_branches: list[str] = field(default_factory=list)
    has_release: bool = True
    has_ci: bool = True

    @property
    def healthy(self) -> bool:
        """True when no issues were found."""
        return not (self.failing_workflows or self.open_branches or not self.has_release or not self.has_ci)

    @property
    def issues_count(self) -> int:
        """Total number of discrete issues found across all checks."""
        return (
            len(self.failing_workflows)
            + len(self.open_branches)
            + (0 if self.has_release else 1)
            + (0 if self.has_ci else 1)
        )

    def stats(self) -> dict[str, int | bool]:
        """Return a summary dict of issue counts."""
        return {
            "failing_workflows": len(self.failing_workflows),
            "open_branches": len(self.open_branches),
            "has_release": self.has_release,
            "has_ci": self.has_ci,
            "issues_count": self.issues_count,
            "healthy": self.healthy,
        }

    def summary_line(self) -> str:
        """Return a single-line human-readable summary."""
        status = "OK" if self.healthy else f"ISSUES ({self.issues_count})"
        return f"{self.name}: {status}"


def _get(url: str, token: str, retries: int = DEFAULT_RETRIES) -> Any:
    """Fetch *url* with Bearer *token*, retrying up to *retries* times on error."""
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r)
        except Exception as exc:
            last_exc = exc
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
    raise RuntimeError(f"All {retries} attempts failed for {url}") from last_exc


def check_repo(name: str, default_branch: str, token: str) -> RepoHealth:
    """Return a RepoHealth for *name*."""
    health = RepoHealth(name=name)
    owner = os.environ.get("GITHUB_USERNAME", "atharvadevne123")

    # Failing workflows
    try:
        runs = _get(
            f"{GH_API}/repos/{owner}/{name}/actions/runs?per_page={WORKFLOW_RUNS_PER_PAGE}",
            token,
        )
        latest: dict[int, Any] = {}
        for run in runs.get("workflow_runs", []):
            wid = run["workflow_id"]
            if wid not in latest:
                latest[wid] = run
        for run in latest.values():
            if run["conclusion"] in FAILING_CONCLUSIONS:
                health.failing_workflows.append(run["name"])
        health.has_ci = bool(latest)
    except Exception as exc:
        log.warning("Could not fetch workflows for %s: %s", name, exc)

    # Open branches
    try:
        branches = _get(f"{GH_API}/repos/{owner}/{name}/branches?per_page={BRANCHES_PER_PAGE}", token)
        health.open_branches = [b["name"] for b in branches if b["name"] != default_branch]
    except Exception as exc:
        log.warning("Could not fetch branches for %s: %s", name, exc)

    # Missing release
    try:
        releases = _get(f"{GH_API}/repos/{owner}/{name}/releases?per_page={RELEASES_PER_PAGE}", token)
        health.has_release = bool(releases)
    except Exception as exc:
        log.warning("Could not fetch releases for %s: %s", name, exc)

    return health


def main() -> int:
    """Entry point. Returns exit code (0 = all healthy)."""
    parser = argparse.ArgumentParser(description="Repo health check")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    token = os.environ.get("GH_PAT", "")
    if not token:
        log.error("GH_PAT environment variable is not set")
        return 1

    owner = os.environ.get("GITHUB_USERNAME", "atharvadevne123")
    repos_data = _get(f"{GH_API}/users/{owner}/repos?per_page=100&type=owner", token)
    repos = [r for r in repos_data if not r.get("archived") and not r.get("fork")]

    results = []
    issues = 0
    for repo in repos:
        health = check_repo(repo["name"], repo.get("default_branch", "main"), token)
        results.append(health)
        if not health.healthy:
            issues += 1

    if args.json:
        import dataclasses

        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
    else:
        for r in results:
            status = "OK" if r.healthy else "ISSUES"
            print(f"[{status}] {r.name}")
            if r.failing_workflows:
                print(f"  Failing workflows: {r.failing_workflows}")
            if r.open_branches:
                print(f"  Open branches: {r.open_branches}")
            if not r.has_release:
                print("  Missing release")
            if not r.has_ci:
                print("  No CI workflows found")
        print(f"\n{len(results)} repos checked, {issues} with issues")

    return 0 if issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
