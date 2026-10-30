# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.2.x   | Yes       |
| 1.1.x   | Yes       |
| < 1.1   | No        |

## Reporting a Vulnerability

To report a security vulnerability, please email devneatharva@gmail.com with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

We aim to respond within 48 hours and will work with you to understand and
address the issue before public disclosure.

## Security Considerations

- The prediction API accepts user-supplied property data. All inputs are
  validated before inference.
- Model files are loaded from the local filesystem at startup. Ensure the
  `MODEL_DIR` path is not writable by untrusted users.
- The `/retrain` endpoint triggers model retraining and should be protected
  behind authentication in production.
- Database credentials must be supplied via environment variables, never
  hardcoded.
