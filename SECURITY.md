# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | yes |

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Open a [security advisory](https://github.com/atharvadevne123/Logistics-Flow/security/advisories/new)
with reproduction steps and the affected version. Expect an initial response
within 72 hours.

## Operational notes

- The rate limiter in `app/middleware.py` is per-process and in-memory. Behind
  multiple replicas it does not enforce a global limit; use Redis or a gateway.
- `DATABASE_URL` and any credentials belong in the environment, never in the
  repository. `.env` is gitignored; `.env.example` holds placeholders only.
- CORS defaults to `allow_origins=["*"]` for local development. Restrict this
  before deploying to production.
