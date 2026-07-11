# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅        |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a GitHub issue with the
`security` label, or email devneatharva@gmail.com for sensitive disclosures.

You can expect an acknowledgement within 72 hours.

## Security Design Notes

- All API inputs are validated with Pydantic v2 (bounds, NaN/Inf rejection,
  length limits on identifiers and batches).
- Database access goes through SQLAlchemy ORM / bound parameters — no string
  SQL interpolation.
- Configuration comes exclusively from environment variables; no secrets are
  committed to the repository (see `.env.example`).
- Rate limiting (sliding window per client IP) protects inference endpoints
  from abuse; configure via `RATE_LIMIT_REQUESTS` and
  `RATE_LIMIT_WINDOW_SECONDS`.
- Validation error responses are sanitized so malformed input (e.g. NaN
  floats) is never echoed back raw.
- Docker image runs a slim Python base with only required build deps.

## Out of Scope

- The bundled `docker-compose.yml` uses example credentials for local
  development only — always override `POSTGRES_PASSWORD` in production.
