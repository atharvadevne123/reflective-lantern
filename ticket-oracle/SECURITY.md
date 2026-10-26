# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | ✓         |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue.
Instead, send details to the maintainer via the repository's security advisory tab
or email the address in `pyproject.toml`.

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- Any suggested mitigation

You will receive an acknowledgement within **48 hours** and a status update within
**7 days**. We aim to release a patch within **30 days** for confirmed issues.

## Security Measures in This Service

- **Input validation**: All request fields validated via Pydantic v2 with explicit
  allow-lists for categorical values.
- **SQL injection**: SQLAlchemy ORM with parameterised queries — raw SQL is never used.
- **Rate limiting**: `slowapi` enforces per-IP request limits (default 60/min).
- **No PII in logs**: Structured logs omit description text; only ticket IDs and
  metadata are recorded.
- **Model artefacts**: Model files are stored locally; no external model registry
  calls are made at inference time.
