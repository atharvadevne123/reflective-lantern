# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 1.0.x | Yes |

## Reporting a vulnerability

Open a private security advisory on the repository rather than a public issue. Include
reproduction steps, affected endpoints, and impact. Expect an acknowledgement within a few
days.

## Operational notes

- **No authentication ships by default.** The API is designed to sit behind a gateway that
  terminates TLS and handles authentication. Do not expose it directly to the internet.
- **Rate limiting is per-process and in-memory.** It protects a single instance from a
  runaway client; it is not a defence against distributed abuse. Use an edge rate limiter
  in production.
- **Secrets come from the environment.** `.env` is gitignored and `.env.example` carries
  placeholders only. Never commit a real `DATABASE_URL`.
- **The container runs as a non-root user** (`appuser`, uid 1001).
- **Prediction payloads are logged to the database.** If seismic station coordinates are
  sensitive in your deployment, restrict access to the `seismic_events` table.
