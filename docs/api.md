# API Reference

## `config.settings`

### `Settings` (frozen dataclass)

Built from environment variables on instantiation.

```python
from config.settings import Settings, get_settings

s = Settings()          # new instance from current env
s = get_settings()      # module-level singleton

missing = s.validate()  # returns list of missing required var names
history = s.history_dir # Path to history/
```

**Fields:**

| Field | Env Var | Default |
|-------|---------|----------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | `""` |
| `gh_pat` | `GH_PAT` | `""` |
| `github_username` | `GITHUB_USERNAME` | `"atharvadevne123"` |
| `notion_api_key` | `NOTION_API_KEY` | `""` |
| `gmail_user` | `GMAIL_USER` | `""` |
| `gmail_app_pass` | `GMAIL_APP_PASS` | `""` |
| `report_recipient` | `REPORT_RECIPIENT` | `gmail_user` |
| `log_level` | `LOG_LEVEL` | `"INFO"` |
| `json_logs` | `JSON_LOGS` | `False` |
| `pf_fix_timeout` | `PF_FIX_TIMEOUT` | `300` |
| `commit_target` | `COMMIT_TARGET` | `60` |

---

## `config.constants`

```python
from config.constants import (
    ROOT_DIR, HISTORY_DIR, SCRIPTS_DIR, COVERS_DIR,
    COMMIT_TARGET, GITHUB_API_BASE, GITHUB_OWNER,
    SMTP_HOST, SMTP_PORT_TLS, SMTP_PORT_SSL,
)
```

---

## `config.logging_config`

```python
from config.logging_config import configure_logging

configure_logging("DEBUG")                # plain-text logs
configure_logging("INFO", json_logs=True) # JSON structured logs
```

---

## `scripts.report_generator`

```python
from datetime import date
from scripts.report_generator import daily_report, weekly_report, load_all_history

report = daily_report(date.today())     # Markdown string
report = weekly_report(date.today())    # Markdown table for past 7 days
history = load_all_history()            # dict[repo_name, list[entry]]
```

---

## `scripts.validate_history`

```python
from pathlib import Path
from scripts.validate_history import validate_file, validate_entry

errors = validate_file(Path("history/MyRepo.json"))  # list[str]
errors = validate_entry(entry_dict, "source", index)  # list[str]
```

---

## `scripts.rotate_repos`

```python
from datetime import date
from scripts.rotate_repos import fetch_repos, select_repo

repos = fetch_repos("atharvadevne123", token)
repo  = select_repo(repos, date.today())  # deterministic selection
print(repo["name"])
```

---

## `scripts.health_check`

```python
from scripts.health_check import check_repo, RepoHealth

health: RepoHealth = check_repo("MyRepo", "main", token)
print(health.healthy)            # bool
print(health.failing_workflows)  # list[str]
print(health.open_branches)      # list[str]
```

---

## `scripts.summarize_history`

```python
from pathlib import Path
from scripts.summarize_history import load_latest_entry

entry = load_latest_entry(Path("history/MyRepo.json"))  # dict | None
```

---

## `scripts.cleanup`

```python
from datetime import date
from scripts.cleanup import clean_file

removed = clean_file(Path("history/MyRepo.json"), cutoff=date(2026, 1, 1))
removed = clean_file(path, cutoff, dry_run=True)  # no file modification
```

---

## `index.js` (Node.js API)

```javascript
const { getSystemPrompt, getPrompts, getHistory, validate } = require('.');

const prompt = getSystemPrompt();         // string
const prompts = getPrompts();             // { [name]: string }
const history = getHistory();             // { [repo]: Array|Object }
const { valid, errors } = validate();     // { valid: bool, errors: string[] }
```
