# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

Please report security vulnerabilities by emailing **devneatharva@gmail.com** with:

1. A description of the vulnerability
2. Steps to reproduce the issue
3. The potential impact
4. Any proposed fix (optional)

**Do not open a public GitHub issue for security vulnerabilities.**

We will acknowledge receipt within 48 hours and aim to release a patch within 14 days for critical issues.

## Security Considerations

### API Authentication
The default deployment does not include API key authentication. For production use, place Forge-Guard behind an API gateway or reverse proxy that enforces authentication.

### Rate Limiting
Built-in rate limiting (default 60 req/min per IP) is configured via `RATE_LIMIT_RPM`. Adjust this for your threat model.

### Input Validation
All sensor readings are validated via Pydantic with strict range constraints. The `/api/v1/predict` endpoint rejects payloads with out-of-range values with HTTP 422.

### Database
- Use PostgreSQL in production (not SQLite)
- Restrict database credentials using the principle of least privilege
- Enable SSL on the database connection via `DATABASE_URL` parameters

### Model Artifacts
- Store `model.joblib` in a secure location with restricted read permissions
- Verify model integrity with checksums before loading in production
- Rotate model artifacts after any suspected tampering

### Environment Variables
Never commit `.env` files. Use the `.env.example` as a template and load secrets via a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.).
