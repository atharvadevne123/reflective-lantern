# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅        |

## Reporting a Vulnerability

To report a security vulnerability, email **devneatharva@gmail.com** with subject `[Signal-Forge Security]`.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

We aim to respond within 48 hours and release a patch within 14 days for confirmed issues.

## Security Practices

- All environment variables (credentials, connection strings) are loaded via `.env` and never hardcoded.
- Database queries use SQLAlchemy parameterised statements — no raw string interpolation.
- API input validated with Pydantic v2 at every endpoint boundary.
- Rate limiting (60 req/min per IP) protects against abuse.
- Dependency updates via Dependabot.
