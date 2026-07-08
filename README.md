![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)
![Python Package](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/python-publish.yml/badge.svg)
![npm](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/npm-publish.yml/badge.svg)
![Bump Version](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/bump-version.yml/badge.svg)

# Reflective Lantern

Autonomous Mon–Fri code improvement agent powered by Claude Code Cloud Routines.

Every weekday at 9 AM CST, Reflective Lantern wakes up, picks one of @atharvadevne123’s
GitHub repositories, implements 60 improvements, runs tests, updates docs, pushes to main,
and sends an email digest — all with zero human intervention.

## Quick Start

```bash
git clone https://github.com/atharvadevne123/reflective-lantern.git
cd reflective-lantern
bash scripts/setup.sh          # install deps + pre-commit hooks
cp .env.example .env           # fill in your API keys
make test                      # verify everything works
```

**Required environment variables** (see `.env.example` for full list):

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for AI-powered features |
| `GH_PAT` | GitHub Personal Access Token (`repo` + `workflow` scopes) |
| `NOTION_API_KEY` | For Notion portfolio updates |
| `GMAIL_USER` + `GMAIL_APP_PASS` | For emailed run reports |

## What It Does

Each daily run:
1. **PRE-FLIGHT** — Fix failing CI workflows, merge open branches, create missing releases
2. **MODE SELECT** — IMPROVEMENT (most days) or INNOVATION (Wednesday wks 2 & 4)
3. **SELECT REPO** — deterministic daily rotation through the active portfolio
4. **ANALYSE** — read every source file, identify 60 improvements across 5 tiers
5. **IMPLEMENT** — one commit per change (security → tests → quality → DX → perf)
6. **VERIFY** — run full test suite; fix failures (2 attempts)
7. **PUSH** — directly to `main`
8. **NOTIFY** — PDF report emailed to devneatharva@gmail.com

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a full ASCII diagram.

```
reflective-lantern/
├── .claude/settings.json     ← CCR tool permissions
├── config/                    ← Settings, constants, logging
├── scripts/                   ← Standalone utility scripts
├── tests/                     ← pytest suite
├── docs/                      ← Architecture & operations docs
├── history/                   ← Per-repo JSON run logs
├── prompts/system_prompt.md  ← Cached agent instructions (3000+ tokens)
└── covers/                    ← SVG cover images for Notion
```

## Improvement Tiers

| Priority | Tier | Examples |
|----------|------|----------|
| 1 | Security / Correctness | Secrets → env vars, bare `except` → typed, SQL injection |
| 2 | Tests | `conftest.py`, happy path + 3 edge cases per endpoint |
| 3 | Code Quality | Type hints, docstrings, logging, refactor > 40-line functions |
| 4 | Developer Experience | CI/CD, Dockerfile, `.env.example`, `pyproject.toml`, README |
| 5 | Performance | `lru_cache`, N+1 fix, DB indexes, connection pooling |

## Utility Scripts

```bash
make health-check          # cross-repo CI / release / branch health
make weekly-summary        # build + email 7-day digest
make validate-history      # validate history JSON schema
make notion-update         # sync Notion portfolio pages
make foundry-export        # export run history as a Foundry-ready CSV
make foundry-sync          # export + upload to a Palantir Foundry dataset

python scripts/summarize_history.py        # tabular run history
python scripts/rotate_repos.py             # which repo is selected today
python scripts/check_ci_status.py --failing-only
python scripts/report_generator.py --mode weekly
```

Foundry setup and dataset schema are documented in [docs/foundry.md](docs/foundry.md).

## Token Efficiency

The `prompts/system_prompt.md` file exceeds Sonnet 4.6’s 2 048-token cache threshold,
so it is cached on first use and subsequent runs hit the cache at ~10% of the original
input cost. Combined with one-repo-per-day rotation, estimated cost is **$0.15–0.25/run**.

## History

The `history/` directory contains JSON logs of every run per repo. Example entry:

```json
{
  "date": "2026-06-30",
  "mode": "improvement",
  "commits": 60,
  "tests_passed": true,
  "improvements": ["added pytest suite", "fixed hardcoded API key"]
}
```

See [`history/schema.json`](history/schema.json) for the full JSON schema.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `make test && make lint` before
opening a pull request.

## License

MIT — see [LICENSE](LICENSE) if present, otherwise open for personal use.

## Tech Stack

- **Scheduler**: Claude Code Cloud Routine (`cron 0 14 * * 1-5` = 9 AM CDT)
- **AI**: Claude Sonnet 4.6 with prompt caching
- **Repo ops**: GitHub REST API + git
- **Notifications**: Gmail SMTP
- **Portfolio**: Notion API + Anthropic SDK
