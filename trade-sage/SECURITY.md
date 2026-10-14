# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

To report a security vulnerability, email the maintainer directly rather than opening a public GitHub issue. Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any proof-of-concept code (kept private until a fix is released)

Expect an initial response within 48 hours and a patch within 14 days for confirmed issues.

## Security Considerations

### API Authentication
The service currently uses IP-based rate limiting (200 req/min). For production deployments, add JWT or API-key authentication via a reverse proxy (e.g., nginx, AWS API Gateway) before exposing the service to the internet.

### Input Validation
All request payloads are validated by Pydantic v2 with strict field constraints:
- OHLCV rows must have `high >= low`
- Portfolio weights must sum to `1.0 ± 0.01`
- Minimum OHLCV sequence lengths are enforced

### Database
- Uses parameterised SQLAlchemy queries — no raw SQL string formatting
- Credentials are loaded from environment variables; never hardcoded
- SQLite is suitable for development only; use PostgreSQL in production

### Dependencies
Pin all dependencies with exact versions in `requirements.txt`. Run `pip audit` regularly to check for known CVEs.

### Docker
The service runs as a non-root `appuser` inside the container. Mount secrets via Docker secrets or environment variables, never bake them into the image.

### CORS
Restrict `CORS_ORIGINS` to your actual frontend domains in production. The default `http://localhost:3000` is for development only.
