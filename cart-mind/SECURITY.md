# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |

## Reporting a vulnerability

Open a private security advisory on the repository rather than a public issue.
Include reproduction steps and the affected endpoint or module.

## Deployment notes

Cart-Mind ships defaults suited to local development. Before exposing it publicly:

- **Database credentials.** `docker-compose.yml` and `.env.example` carry placeholder
  credentials (`cartmind`/`secret`). Replace them; never deploy the defaults.
- **Rate limiting.** The built-in limiter is per-process and in-memory, so with multiple
  workers or replicas the effective limit is `RATE_LIMIT × instances`. Put a shared
  limiter (gateway or Redis-backed) in front for a real enforcement boundary.
- **Authentication.** There is none. Every endpoint is unauthenticated by design —
  Cart-Mind expects to sit behind a gateway that handles authn/authz.
- **Prediction logs.** `prediction_logs` stores user IDs against scored items. Treat the
  table as personal data: apply your retention policy and restrict access.
- **CORS.** No CORS middleware is configured, so browsers block cross-origin calls by
  default. Add `CORSMiddleware` with an explicit allowlist if you need browser access —
  do not use `allow_origins=["*"]` on an authenticated deployment.
