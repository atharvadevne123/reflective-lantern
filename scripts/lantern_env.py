"""Runtime capability detection for Reflective Lantern.

Reflective Lantern runs in more than one execution context — a Claude Code web
sandbox, a GitHub Actions runner, or a developer's machine — and those contexts
differ in ways that silently break the routine:

- The web sandbox intercepts ``api.github.com`` and **discards** the caller's
  ``Authorization`` header, answering with its own session identity instead.
  A hardcoded PAT is therefore not merely unnecessary there, it is inert.
- That session identity is a GitHub App installation token, which cannot
  enumerate an account's repositories and cannot create new ones.
- Outbound SMTP is unreachable when egress is restricted to an HTTPS proxy.

Rather than hardcoding assumptions about any one of these, the routine probes
for them once per run and adapts. Every probe is read-only and side-effect
free; see :func:`can_create_repo` for the one case that needs care.

Usage:
    from scripts.lantern_env import detect

    caps = detect()
    if caps.can_enumerate_repos:
        ...run the pre-flight sweeps...
    else:
        ...skip them and say so in the report...

Run directly for a human-readable summary::

    python3 scripts/lantern_env.py
"""

from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
OWNER = os.environ.get("LANTERN_GITHUB_OWNER", "atharvadevne123")

# A syntactically valid but certainly-invalid token. Used to tell an
# intercepting proxy (which ignores the header) apart from real GitHub (which
# rejects it with 401).
_SENTINEL_TOKEN = "ghp_0000000000000000000000000000000000000"

_TIMEOUT = 12


@dataclass
class Capabilities:
    """What the current execution context actually permits.

    Attributes:
        credentials_honoured: True if a supplied token reaches GitHub. When
            False, the environment injects its own identity and any configured
            PAT is inert.
        can_enumerate_repos: True if the account's repositories can be listed,
            which the pre-flight sweeps require.
        can_create_repo: True if new repositories can be created, which
            innovation mode requires for a standalone project repo.
        can_push_git: True if git push over HTTPS works.
        smtp_reachable: True if an SMTP port accepts a TCP connection.
        notes: Human-readable findings, one per probe.
    """

    credentials_honoured: bool = False
    can_enumerate_repos: bool = False
    can_create_repo: bool = False
    can_push_git: bool = False
    smtp_reachable: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the capability set as a plain dict."""
        return asdict(self)

    def summary(self) -> str:
        """Return a multi-line human-readable summary."""

        def mark(v: bool) -> str:
            return "yes" if v else "NO"

        lines = [
            "Reflective Lantern — runtime capabilities",
            "",
            f"  credentials honoured : {mark(self.credentials_honoured)}",
            f"  enumerate repos      : {mark(self.can_enumerate_repos)}",
            f"  create repositories  : {mark(self.can_create_repo)}",
            f"  git push             : {mark(self.can_push_git)}",
            f"  SMTP reachable       : {mark(self.smtp_reachable)}",
            "",
            "Findings:",
        ]
        lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


def _request(path: str, token: str | None = None, method: str = "GET", payload: dict | None = None) -> tuple[int, str]:
    """Issue a GitHub API request and return (status_code, body).

    Args:
        path: API path beginning with a slash.
        token: Optional bearer token to send.
        method: HTTP method.
        payload: Optional JSON body.

    Returns:
        Tuple of HTTP status code and response body text. A status of 0
        indicates the request could not be completed at all.
    """
    url = f"{GITHUB_API}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:
        logger.debug("Request to %s failed: %s", path, exc)
        return 0, str(exc)


def credentials_are_honoured() -> tuple[bool, str]:
    """Detect whether a supplied token actually reaches GitHub.

    Sends a deliberately invalid token to an endpoint that requires
    authentication. Real GitHub answers 401. An intercepting proxy that
    substitutes its own identity answers 200 — proving the caller's token is
    discarded and any configured PAT is inert.

    Returns:
        Tuple of (honoured, explanatory note).
    """
    status, _ = _request("/user", token=_SENTINEL_TOKEN)
    if status == 401:
        return True, "Credentials are honoured: an invalid token was correctly rejected."
    if status == 200:
        return False, (
            "Credentials are IGNORED: an invalid token still returned 200, so the "
            "environment injects its own identity. A configured PAT is inert here."
        )
    return False, f"Credential handling indeterminate (HTTP {status} for an invalid token)."


def can_enumerate_repos() -> tuple[bool, str]:
    """Detect whether the account's repositories can be listed.

    Returns:
        Tuple of (allowed, explanatory note).
    """
    status, body = _request(f"/users/{OWNER}/repos?per_page=1&type=owner")
    if status == 200:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return False, "Repo enumeration returned 200 but the body was not JSON."
        if isinstance(parsed, list):
            return True, "Repo enumeration works — pre-flight sweeps can run."
        message = parsed.get("message", "")
        if "bound to their configured repositories" in message:
            return False, (
                "Repo enumeration BLOCKED: the session is bound to its configured "
                "repositories. Pre-flight sweeps cannot run in this context."
            )
        return False, f"Repo enumeration blocked: {message[:120]}"
    return False, f"Repo enumeration blocked (HTTP {status})."


def can_create_repo() -> tuple[bool, str]:
    """Detect whether new repositories can be created, without creating one.

    Posts an intentionally invalid repository name. A context that permits
    creation rejects it with 422 (validation failed); one that forbids
    creation rejects it with 403 before validation. Neither outcome creates a
    repository.

    Returns:
        Tuple of (allowed, explanatory note).
    """
    status, body = _request("/user/repos", method="POST", payload={"name": ""})
    if status == 422:
        return True, "Repository creation is permitted (validation reached)."
    if status == 403:
        return False, (
            "Repository creation FORBIDDEN (403). GitHub App installation tokens "
            "cannot create repositories on a personal account. Build inside an "
            "existing repository instead."
        )
    if status in (401, 404):
        return False, f"Repository creation unavailable (HTTP {status})."
    if "bound to their configured repositories" in body:
        return False, ("Repository creation BLOCKED: the session is bound to its configured repositories.")
    return False, f"Repository creation unavailable (HTTP {status})."


def can_push_git(remote: str | None = None) -> tuple[bool, str]:
    """Detect whether git can reach the origin remote.

    Args:
        remote: Optional explicit remote URL. Defaults to the repo's origin.

    Returns:
        Tuple of (reachable, explanatory note).
    """
    import subprocess

    cmd = ["git", "ls-remote", "--heads"]
    if remote:
        cmd.append(remote)
    else:
        cmd.append("origin")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=45, text=True)
    except Exception as exc:
        return False, f"git ls-remote failed to run: {exc}"

    if result.returncode == 0:
        return True, "git reaches origin — commits can be pushed."
    return False, f"git cannot reach origin: {result.stderr.strip()[:140]}"


def smtp_reachable(
    hosts: tuple[tuple[str, int], ...] = (
        ("smtp.gmail.com", 587),
        ("smtp.gmail.com", 465),
    ),
) -> tuple[bool, str]:
    """Detect whether any SMTP endpoint accepts a TCP connection.

    Args:
        hosts: Tuples of (hostname, port) to try in order.

    Returns:
        Tuple of (reachable, explanatory note).
    """
    for host, port in hosts:
        try:
            with socket.create_connection((host, port), timeout=10):
                return True, f"SMTP reachable at {host}:{port}."
        except Exception as exc:
            logger.debug("SMTP %s:%s unreachable: %s", host, port, exc)
    return False, (
        "SMTP UNREACHABLE on all ports — egress is HTTPS-proxy-only. Email "
        "delivery is impossible; write the report into the repository instead."
    )


def detect(use_cache: bool = True, cache_path: str | None = None) -> Capabilities:
    """Probe the execution context and return its capabilities.

    Args:
        use_cache: If True, reuse a cached result from earlier in the same run.
        cache_path: Override for the cache file location.

    Returns:
        A populated Capabilities instance.
    """
    path = Path(cache_path or os.environ.get("LANTERN_CAPS_CACHE", "/tmp/lantern_capabilities.json"))

    if use_cache and path.exists():
        try:
            return Capabilities(**json.loads(path.read_text()))
        except Exception:
            logger.debug("Capability cache unreadable — re-probing")

    caps = Capabilities()
    for attr, probe in (
        ("credentials_honoured", credentials_are_honoured),
        ("can_enumerate_repos", can_enumerate_repos),
        ("can_create_repo", can_create_repo),
        ("can_push_git", can_push_git),
        ("smtp_reachable", smtp_reachable),
    ):
        ok, note = probe()
        setattr(caps, attr, ok)
        caps.notes.append(note)
        logger.info("%s: %s", attr, note)

    try:
        path.write_text(json.dumps(caps.to_dict(), indent=2))
    except OSError:
        logger.debug("Could not write capability cache to %s", path)

    return caps


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    print(detect(use_cache=False).summary())
