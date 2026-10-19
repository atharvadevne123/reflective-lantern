# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅        |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a **private** GitHub Security Advisory
(Security → Advisories → New draft advisory) rather than a public issue.

Include:
- Description of the vulnerability and its potential impact
- Steps to reproduce
- Suggested fix (if available)

We aim to acknowledge reports within 72 hours and release a patch within 14 days for critical issues.

## Security Considerations

- The `/train` endpoint should be protected with authentication in production.
- The `DATABASE_URL` must never be logged or exposed via API responses.
- Rate limiting is enabled by default (60 req/min per IP); adjust `RATE_LIMIT` for your use case.
- Always run behind a reverse proxy (nginx/caddy) in production.
