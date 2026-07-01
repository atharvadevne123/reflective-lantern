# Operations Runbook

## Daily Routine

Reflective Lantern runs automatically at **9 AM CST Monday–Friday** via a
Claude Code Cloud Routine. No manual intervention is required under normal
conditions.

## Monitoring

### Check run history
```bash
python scripts/summarize_history.py
```

### Check CI health across all repos
```bash
export GH_PAT=ghp_...
python scripts/health_check.py
```

### View failing workflows
```bash
python scripts/check_ci_status.py --failing-only
```

## Alerts

A PushNotification is sent when:
- The run surfaces a condition the routine was watching for
- A CI failure is found that needs attention
- The email send succeeded or failed

No notification is sent when everything is healthy (silence = OK).

## Manual Triggers

### Run notion portfolio update
```bash
make notion-update
# or with AI-generated descriptions:
make notion-update-descriptions
```

### Generate a weekly summary
```bash
make weekly-summary
# or dry-run (print without sending):
python scripts/generate_weekly_summary.py --dry-run
```

### Validate history files
```bash
make validate-history
```

### Clean old history entries (older than 90 days)
```bash
python scripts/cleanup.py --days 90
# Preview without modifying:
python scripts/cleanup.py --days 90 --dry-run
```

## Troubleshooting

### Routine didn't run
1. Check the Cloud Routine cron schedule in `.claude/settings.json`
2. Verify `GH_PAT` has `repo` + `workflow` scopes
3. Check Claude Code Cloud Routine logs

### Email not received
1. Verify `GMAIL_USER` and `GMAIL_APP_PASS` env vars
2. Check spam/junk folder
3. Run `python scripts/generate_weekly_summary.py --dry-run` to validate

### Commits < 60
The commit gate in Phase 7.5 logs the shortfall. Check the run transcript
for which tier ran out of improvements. Consider adding more repos or
explicitly adding documentation / test expansion passes.

### CI failing in target repo
The pre-flight (Phase 1) automatically applies ruff fixes. If the fix is
complex (broken test logic, missing dependency), a TODO comment is left
and the routine moves on. Fix manually and re-run.

## Environment Variables

See `.env.example` for the full list with descriptions.

Required at runtime:
- `ANTHROPIC_API_KEY`
- `GH_PAT`
- `NOTION_API_KEY` (for notion update scripts)
- `GMAIL_USER` + `GMAIL_APP_PASS` (for email reports)
