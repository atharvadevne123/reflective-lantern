# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅        |

## Reporting a vulnerability

Report security issues privately to **devneatharva@gmail.com** rather than
opening a public issue. Please include:

- A description of the vulnerability and its impact
- Steps to reproduce, or a proof-of-concept
- Affected version and deployment configuration

You can expect an acknowledgement within 72 hours and a status update within
seven days.

## Security posture

Ops-Vision is designed to run behind an authenticated gateway. Note the
following when deploying:

- **No built-in authentication.** The API assumes network-level or gateway
  authentication. Do not expose it directly to the public internet.
- **Rate limiting is per-process and in-memory.** `RateLimitMiddleware` keys on
  client IP and does not share state across replicas, so a multi-replica
  deployment enforces the limit per replica, not globally. Use a shared limiter
  at the gateway when a global limit matters.
- **`X-Forwarded-For` is trusted.** The limiter reads the first address in the
  chain. Only run behind a proxy that overwrites this header, otherwise a
  client can spoof it to evade throttling.
- **Model artifacts are unpickled.** `load_model()` uses `pickle`, which
  executes arbitrary code on load. Only load model files from trusted storage
  you control.
- **Secrets come from the environment.** No credential is committed to the
  repository; `.env` is git-ignored and `.env.example` holds placeholders only.
- **Container runs as non-root.** The image creates and switches to `appuser`
  (UID 1001).

## Dependency management

Runtime dependencies are pinned in `requirements.txt`. CI runs on every push
and pull request against the default branch.
