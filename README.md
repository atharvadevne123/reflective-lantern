# Reflective Lantern

Autonomous Mon–Fri code improvement agent powered by Claude Code Cloud Routines.

Every weekday at 9 AM CST, Reflective Lantern wakes up, picks one of @atharvadevne123's
GitHub repositories, implements 5+ improvements, runs tests, updates docs, pushes to main,
and sends an email digest — all with zero human intervention.

## What It Does

Each daily run:
1. **Discovers** all active GitHub repos (skips archived/forks)
2. **Rotates** through repos on a deterministic weekly schedule
3. **Analyzes** the codebase for improvement opportunities
4. **Implements** 5+ changes across these priority tiers:
   - Security & correctness (hardcoded secrets, missing error handling)
   - Test coverage (creates pytest/jest suites from scratch if needed)
   - Code quality (type hints, docstrings, logging)
   - Developer experience (CI/CD, .env.example, Dockerfile)
   - Performance (caching, N+1 queries)
5. **Tests** all changes before pushing
6. **Pushes** directly to `main`
7. **Notifies** via Gmail digest to devneatharva@gmail.com

## Structure

```
reflective-lantern/
├── .claude/settings.json     ← CCR tool permissions
├── prompts/system_prompt.md  ← cached agent instructions (3000+ tokens)
├── history/                  ← per-repo JSON run logs
└── README.md
```

## Tech Stack

- **Scheduler**: Claude Code Cloud Routine (cron `0 14 * * 1-5` = 9 AM CDT)
- **AI**: Claude Sonnet 4.6 with prompt caching (~80% token reduction)
- **Repo ops**: GitHub CLI (`gh`) + git
- **Notifications**: Gmail MCP

## Token Efficiency

The `prompts/system_prompt.md` file exceeds Sonnet 4.6's 2048-token cache threshold,
so it is cached on first use and subsequent runs hit the cache at ~10% of the original
input cost. Combined with one-repo-per-day rotation, estimated cost is $0.15–0.25/run.

## History

The `history/` directory contains JSON logs of every run per repo, used to avoid
repeating improvements across runs. Example entry:

```json
[
  {
    "date": "2026-04-21",
    "improvements": ["added pytest suite", "fixed hardcoded API key"],
    "tests_passed": true,
    "commits": 3,
    "notes": ""
  }
]
```
