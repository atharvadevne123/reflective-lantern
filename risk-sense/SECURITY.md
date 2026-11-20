# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

To report a security vulnerability, please email the maintainers directly rather than opening a public issue.

**Do not** disclose security issues publicly until they have been addressed.

## Security Considerations

### API Rate Limiting

The `/api/v1/score` endpoint enforces a rate limit of 100 requests per 60 seconds per IP address. Requests exceeding this limit return HTTP 429.

### Data Handling

- Applicant data is stored in a PostgreSQL database (SQLite in development).
- Database credentials are loaded from environment variables only; never hardcoded.
- The `DATABASE_URL` is excluded from the `/api/v1/metrics` response.

### Input Validation

All inputs are validated via Pydantic v2 schemas with strict type coercion. Fields are range-checked:
- `annual_income`: must be positive
- `debt_to_income`: must be in [0, 1]
- `credit_utilization`: must be in [0, 1]
- `num_derogatory_marks`, `num_credit_inquiries`, `num_open_accounts`: must be non-negative integers

### Model Security

- Pickle serialisation is used for model persistence. Load models only from trusted paths controlled by the deployment environment.
- The `MODEL_PATH` and `PIPELINE_PATH` environment variables must resolve to paths that are not user-writable in production.

### Dependencies

Dependencies are pinned in `requirements.txt`. Run `pip-audit` before each release to check for known CVEs.
