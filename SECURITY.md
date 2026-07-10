# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x.x   | Yes       |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately to **devneatharva@gmail.com** with:

1. A description of the vulnerability
2. Steps to reproduce (minimal proof-of-concept if possible)
3. Potential impact and affected versions
4. Your suggested fix, if any

You will receive an acknowledgement within **48 hours**. Confirmed
vulnerabilities will be patched and released promptly. We will credit
reporters in the release notes unless you prefer anonymity.

## Security Considerations for Contributors

- **Never commit API keys, tokens, or passwords** — use environment variables
  and reference `.env.example`
- The `detect-private-key` pre-commit hook will block accidental secret commits
- `GH_PAT`, `ANTHROPIC_API_KEY`, `NOTION_API_KEY`, and `GMAIL_APP_PASS` are
  runtime secrets managed via environment variables only
- All GitHub API calls use HTTPS with token-based authentication
- SMTP connections use TLS (port 587 with STARTTLS or port 465 with SSL)

## Threat Model

Reflective Lantern is a scheduled automation agent. The primary attack surfaces are:

| Surface | Mitigation |
|---------|------------|
| Leaked `GH_PAT` | Scoped to `repo` + `workflow` only; rotate immediately if exposed |
| Injected content from repo files | Agent reads but does not execute arbitrary file content |
| SMTP credentials | Gmail App Password (not account password); revoke in Google account settings |
| Notion API key | Read/write scoped to specific database; revoke in Notion integrations |

## Dependency Updates

Dependabot is configured to open weekly PRs for npm, pip, and GitHub Actions
dependency updates. Review and merge these promptly.
