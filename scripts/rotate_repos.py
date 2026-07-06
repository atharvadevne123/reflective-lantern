#!/usr/bin/env python3
"""Deterministic daily repo selection for Reflective Lantern.

Given today's date seed, picks one repo from the owner's active repo list,
excluding archived repos, forks, and reflective-lantern itself.

Usage:
    python scripts/rotate_repos.py
    python scripts/rotate_repos.py --date 2026-07-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import urllib.request
from datetime import date
from typing import Any

log = logging.getLogger(__name__)
GH_API = "https://api.github.com"


def fetch_repos(owner: str, token: str) -> list[dict[str, Any]]:
    """Return all active, non-fork repos for *owner*."""
    url = f"{GH_API}/users/{owner}/repos?per_page=100&type=owner"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        repos = json.load(r)
    return [
        r for r in repos
        if not r.get("archived")
        and not r.get("fork")
        and r["name"] != "reflective-lantern"
    ]


def select_repo(
    repos: list[dict[str, Any]],
    target_date: date,
    min_repos: int = 1,
) -> dict[str, Any]:
    """Select a repo deterministically based on *target_date*.

    Raises ValueError when the repo list has fewer than *min_repos* entries.
    """
    if len(repos) < min_repos:
        raise ValueError(
            f"Need at least {min_repos} repo(s), got {len(repos)}"
        )
    rng = random.Random(target_date.year * 10000 + target_date.month * 100 + target_date.day)
    return rng.choice(repos)


def main() -> int:
    """Entry point. Prints the selected repo name."""
    parser = argparse.ArgumentParser(description="Select today's repo for improvement")
    parser.add_argument("--date", help="Override date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output full repo metadata as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    token = os.environ.get("GH_PAT", "")
    if not token:
        log.error("GH_PAT is not set")
        return 1

    owner = os.environ.get("GITHUB_USERNAME", "atharvadevne123")
    target_date = date.fromisoformat(args.date) if args.date else date.today()

    repos = fetch_repos(owner, token)
    if not repos:
        log.error("No eligible repos found for %s", owner)
        return 1

    repo = select_repo(repos, target_date)

    if args.json:
        print(json.dumps({
            "name": repo["name"],
            "language": repo.get("language"),
            "description": repo.get("description"),
            "date": target_date.isoformat(),
        }, indent=2))
    else:
        print(repo["name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
