# Architecture

## Overview

Reflective Lantern is a scheduled autonomous agent that runs every weekday at 9 AM CST
using Claude Code Cloud Routines. Its sole job is to keep every repository in
`@atharvadevne123`'s portfolio continuously improving.

```
                    ┌──────────────────────────────────┐
                    │   Claude Code Cloud Routine       │
                    │   cron: 0 14 * * 1-5 (9 AM CDT)  │
                    └────────────────┬─────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │        reflective-lantern         │
                    │                                   │
                    │  prompts/system_prompt.md         │
                    │  history/<repo>.json              │
                    └────────────────┬─────────────────┘
                                     │
           ┌─────────────────────────▼────────────────────────┐
           │                  PRE-FLIGHT                       │
           │  1. Fix failing CI across all repos               │
           │  2. Merge non-main branches                       │
           │  3. Create missing releases                        │
           └─────────────────────────┬────────────────────────┘
                                     │
           ┌─────────────────────────▼────────────────────────┐
           │              MODE SELECTION (PHASE 0)             │
           │  Wednesday days 8-14 or 22-28 → INNOVATION        │
           │  All other weekdays → IMPROVEMENT                 │
           └─────────────────────────┬────────────────────────┘
                                     │
        ┌────────────────────────────┴────────────────────────┐
        │                                                      │
┌───────▼────────┐                               ┌───────────▼────────┐
│  IMPROVEMENT   │                               │   INNOVATION MODE  │
│                │                               │                    │
│ 1. Select repo │                               │ 1. Scrape HN       │
│ 2. Clone       │                               │ 2. Pick ML idea    │
│ 3. Orient      │                               │ 3. Build from      │
│ 4. Plan 60     │                               │    scratch (60     │
│    commits     │                               │    commits)        │
│ 5. Implement   │                               │ 4. Release + wheel │
│ 6. Test        │                               └───────────┬────────┘
│ 7. Push main   │                                           │
└───────┬────────┘                                           │
        └───────────────────────┬───────────────────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │  LOG + EMAIL (PHASE 9/D)      │
                 │  history/<repo>.json updated  │
                 │  PDF report emailed            │
                 │  Push to reflective-lantern   │
                 └──────────────────────────────┘
```

## Components

### `prompts/system_prompt.md`
The master instruction set (3 000+ tokens). Exceeds Sonnet 4.6's 2 048-token
prompt-cache threshold, so it is cached on first use and subsequent runs hit
the cache at ~10% of original cost.

### `history/`
One JSON file per repo. Tracks every run's date, mode, commit count, improvements,
and test status. Used to avoid repeating improvements across runs.

### `scripts/`
Utility scripts used by the agent and callable standalone:

| Script | Purpose |
|--------|---------|
| `notion_portfolio_update.py` | Sync Notion portfolio case studies |
| `health_check.py` | Check CI / release / branch health across all repos |
| `report_generator.py` | Generate daily/weekly Markdown reports |
| `validate_history.py` | Validate history JSON schema |
| `summarize_history.py` | Print tabular summary of all runs |
| `cleanup.py` | Remove old history entries |
| `rotate_repos.py` | Deterministic daily repo selection |
| `check_ci_status.py` | Report CI status across all repos |
| `generate_weekly_summary.py` | Build and email weekly digest |

### `config/`
Pure-Python configuration package:
- `constants.py` — directory paths, limits, API base URLs
- `settings.py` — `Settings` dataclass built from env vars, with `validate()`
- `logging_config.py` — `configure_logging()` supporting JSON log output

## Token Efficiency

- Prompt caching reduces per-run input cost by ~80%
- `glob` before `read` for file discovery
- `pytest 2>&1 | tail -60` to avoid flooding the context window
- Never reads `node_modules/`, `venv/`, `__pycache__/`, `.git/`
