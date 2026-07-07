# Changelog

All notable changes to Reflective Lantern are documented here.

Format: [Semantic Versioning](https://semver.org/) —
`[version] — YYYY-MM-DD`

---

## [Unreleased]

### Added
- `config/` package: `Settings`, `constants`, `logging_config`
- `scripts/health_check.py` — cross-repo CI/release/branch health check
- `scripts/report_generator.py` — daily and weekly Markdown reports
- `scripts/validate_history.py` — JSON schema validation for history files
- `scripts/summarize_history.py` — tabular run history summary
- `scripts/cleanup.py` — remove old history entries
- `scripts/rotate_repos.py` — deterministic daily repo selection
- `scripts/check_ci_status.py` — across-repo CI status reporter
- `scripts/generate_weekly_summary.py` — weekly digest emailer
- `tests/` — full pytest suite with fixtures and parametrized tests
- `.github/workflows/ci.yml` — lint, test, type-check on every push
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates
- `.github/PULL_REQUEST_TEMPLATE.md`
- `docs/architecture.md`, `docs/operations.md`
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- `pyproject.toml`, `.pre-commit-config.yaml`, `Makefile`, `mypy.ini`
- `.env.example`, `.gitignore`, `.nvmrc`

### Changed
- `scripts/notion_portfolio_update.py` — added type annotations, structured
  logging, retry logic, and input validation
- `index.js` — added JSDoc, error handling, `getHistory()`, `validate()`
- `package.json` — added `test`, `lint`, `validate` scripts
- `README.md` — expanded with Quick Start, Architecture, API Reference,
  Contributing, and Examples sections

---

## [1.0.0] — 2026-04-21

### Added
- Initial release: daily autonomous code improvement agent
- Per-repo JSON history tracking
- Gmail PDF report emails
- IMPROVEMENT and INNOVATION modes
- Notion portfolio updater
