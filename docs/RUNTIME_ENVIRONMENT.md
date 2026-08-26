# Runtime environment constraints

Reflective Lantern's routine prompt was written assuming an unrestricted GitHub
PAT and outbound SMTP. When it runs in a Claude Code **web sandbox**, neither
assumption holds. This document records what was measured, so future runs adapt
instead of rediscovering it — and so nobody spends another run debugging
credentials that were never the problem.

Detect the current context at any time:

```bash
python3 scripts/lantern_env.py
```

## Measured on 2026-08-26 (Claude Code web sandbox)

| Capability | Status | Consequence |
|---|---|---|
| Credentials honoured | **No** | A configured `GH_PAT` is inert |
| Enumerate repositories | **No** | Pre-flight sweeps cannot run |
| Create repositories | **No** | Innovation mode cannot make a standalone repo |
| Git push | Yes | Commits reach `main` normally |
| SMTP reachable | **No** | Email reports are impossible |

### Credentials are discarded, not rejected

`api.github.com` is intercepted by the egress proxy, which **drops the
`Authorization` header** and answers with the session's own identity. The
decisive evidence:

| Request | Result |
|---|---|
| `GET /user` with the real PAT | `200` |
| `GET /user` with `ghp_TOTALLY_INVALID…` | `200` |
| `GET /user` with no token at all | `200` |
| `git ls-remote` with a garbage token / no creds | both succeed |

An invalid token returning `200` proves the caller's credential is never
evaluated. Putting a PAT in the routine prompt therefore buys nothing — it is
purely a credential-exposure risk. `scripts/lantern_env.py` probes for exactly
this by sending a deliberately invalid token and checking for a `401`.

### The session identity is a GitHub App installation token

That identity is scoped to the repositories configured on the environment, and
carries two hard limits:

- **No enumeration.** `GET /user/repos` is refused by the proxy with
  *"sessions are bound to their configured repositories."* All three pre-flight
  sweeps (failing CI, stray branches, missing releases) depend on it.
- **No repository creation.** `POST /user/repos` returns
  `403 Resource not accessible by integration`. This is a GitHub limitation:
  installation tokens have no equivalent of user-account repo creation. It is
  **not** fixable by granting the App more permissions.

> Granting the Claude GitHub App "All repositories" on 2026-08-26 did **not**
> lift either limit. The App installation and the session's repo binding are
> separate layers; the refusals above come from the session binding, which is
> set by the Claude Code environment configuration, not by GitHub.

### SMTP is unreachable

Ports 587 and 465 fail at TCP connect — egress is HTTPS-proxy-only. Email
delivery cannot work here by any configuration of credentials.

## How the routine adapts

`scripts/lantern_env.py` probes each capability once per run and returns a
`Capabilities` object. Every probe is read-only; the repo-creation probe posts
an intentionally **empty** repository name, so a permitted context answers
`422` (validation reached) and a forbidden one answers `403` — neither creates
anything.

```python
from scripts.lantern_env import detect

caps = detect()
if caps.can_enumerate_repos:
    run_preflight_sweeps()
else:
    note_skipped("repo enumeration unavailable in this context")
```

`scripts/send_report.py` follows the same principle: it tries email, and when
that is unavailable it files the report under `reports/` so the run still
produces a durable artifact. Delivery never raises — a run must not fail
because its reporting channel is down.

## Getting the full routine working

The limits above are properties of the execution context, not bugs to be fixed
in this repository. To run the routine as originally written:

1. **Move execution to where credentials are honoured** — GitHub Actions with
   `GH_PAT` as a repository secret, or the Claude Code CLI on a local machine.
   This restores enumeration *and* repository creation in one move.
2. **Or widen the environment's repo binding** — see
   <https://code.claude.com/docs/en/claude-code-on-the-web>. This may restore
   the sweeps. It will **not** restore repository creation.
3. **Or accept the monorepo layout** — build each project as a subdirectory,
   as `ops-vision/` does. This needs no infrastructure change and is a
   perfectly reasonable end state.

## Credential hygiene

Secrets must come from the environment, never from source. `send_report.py`
reads `LANTERN_SMTP_PASSWORD`, `LANTERN_REPORT_TO`, `LANTERN_SMTP_USER`, and
`LANTERN_SMTP_HOST`; when the password is unset it skips email and files the
report instead.

> **Historical exposure.** A Gmail app password was previously committed in
> plaintext to `scripts/send_report.py` and
> `history/pending/watt_guard_report_2026-07-11.txt`. Both have been redacted,
> but **redaction is not revocation** — the value remains in git history and in
> any clone or fork. That credential must be revoked at
> <https://myaccount.google.com/apppasswords>. The GitHub PAT carried in the
> routine prompt was never committed, but should also be revoked: it is inert
> in this environment and is a liability wherever it is stored.
