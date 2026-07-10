# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | Yes       |

## Reporting a Vulnerability

Please report security vulnerabilities to devneatharva@gmail.com.
Do not open public GitHub issues for security vulnerabilities.

## Security Best Practices

- Use environment variables for all secrets (see `.env.example`)
- Do not expose the `/api/v1/retrain` endpoint publicly without authentication
- Use HTTPS in production (configure via reverse proxy, e.g., nginx)
- PostgreSQL credentials should be rotated regularly
