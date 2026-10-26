# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | yes |

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Open a [security advisory](https://github.com/atharvadevne123/Logistics-Flow/security/advisories/new)
with reproduction steps and the affected version. Expect an initial response
within 72 hours.

### Responsible disclosure policy

We ask that you:

1. **Do not** publicly disclose the vulnerability before a fix is available.
2. Provide sufficient detail to reproduce the issue (steps, environment,
   payload, and expected vs actual behaviour).
3. Allow us a reasonable time (up to 90 days) to investigate, confirm, and
   deploy a fix before any public disclosure.
4. Avoid accessing or modifying data that does not belong to you while
   researching the issue.

In return we commit to:

- Acknowledge your report within 72 hours.
- Keep you informed of our progress.
- Credit you in the release notes unless you prefer to remain anonymous.
- Not pursue legal action against researchers acting in good faith under this
  policy.

## Operational notes

- The rate limiter in `app/middleware.py` is per-process and in-memory. Behind
  multiple replicas it does not enforce a global limit; use Redis or a gateway.
- `DATABASE_URL` and any credentials belong in the environment, never in the
  repository. `.env` is gitignored; `.env.example` holds placeholders only.
- CORS defaults to `allow_origins=["*"]` for local development. Restrict this
  before deploying to production.
- `SECRET_KEY` in `.env.example` defaults to `change-me-in-production`; override
  it with a strong random value before any deployment.
- SQL queries use SQLAlchemy parameterized statements throughout; do not
  interpolate user input into raw queries.
- All prediction inputs are validated by Pydantic models before processing;
  unknown carriers and route types are rejected at the API boundary.

## Dependency management

- Pin all production dependencies to a minor version range in `requirements.txt`
  and review updates with `pip-audit` before merging.
- The `ruff` linter enforces `B` (bugbear) rules to catch common security-adjacent
  mistakes at lint time.
