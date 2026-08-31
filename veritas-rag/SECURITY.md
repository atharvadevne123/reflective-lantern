# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅ Yes    |

## Reporting a Vulnerability

If you discover a security vulnerability in Veritas-RAG, please **do not** open a public
GitHub issue. Instead, email **devneatharva@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Any suggested mitigations

You will receive a response within 48 hours. We will work with you to understand and
address the issue before any public disclosure.

## Security Considerations

- All API keys and credentials must be supplied via environment variables (never hardcoded)
- Database connection strings must be stored in `.env` (excluded from version control)
- Input validation is enforced via Pydantic schemas on all endpoints
- SQL queries use SQLAlchemy parameterised statements to prevent injection
