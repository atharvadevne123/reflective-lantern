# Palantir Foundry Integration

Reflective Lantern can export its run history as a Foundry-ready dataset and
push it to a Foundry stack using the Datasets v2 REST API.

## Components

| Module | Purpose |
|---|---|
| `scripts/foundry_export.py` | Flatten `history/*.json` into a tabular dataset (CSV / JSONL) or Ontology object payloads |
| `scripts/foundry_client.py` | Transactional upload client (create → upload → commit, abort on failure) |
| `scripts/foundry_sync.py` | One-command export + upload with graceful export-only fallback |

## Dataset schema

Every history entry becomes one row with a stable column order:

| Column | Type | Notes |
|---|---|---|
| `run_key` | string | `<repo>:<date>` — primary key, safe for upserts |
| `repo` | string | repository name (history filename stem) |
| `date` | string | ISO date of the run (`last_run` fallback honoured) |
| `mode` | string | `improvement` / `innovation`, lowercased |
| `commits` | int | commits made in the run |
| `tests_passed` | bool | whether the suite was green |
| `improvement_count` | int | number of recorded improvements |
| `email_status` | string | empty when the entry stores a non-string status |

The `--format ontology` mode instead emits one `Repository` object per repo
(with `totalRuns` / `totalCommits` aggregates) and one `AutomationRun` object
per row, shaped for Foundry Ontology ingestion.

## Setup

1. **Hostname** — your stack URL, e.g. `https://yourstack.palantirfoundry.com`.
2. **Token** — in Foundry: *Account settings → Tokens → generate token*.
3. **Dataset RID** — create an empty dataset in a project folder and copy its
   RID (`ri.foundry.main.dataset.…`) from the URL or About panel.

Set the environment variables (locally via `.env`, or in your Claude Code
environment settings for cloud runs):

```bash
FOUNDRY_HOSTNAME=https://yourstack.palantirfoundry.com
FOUNDRY_TOKEN=eyJ...            # keep secret
FOUNDRY_DATASET_RID=ri.foundry.main.dataset.xxxx
FOUNDRY_BRANCH=master           # optional, defaults to master
```

If the runtime has a restricted network policy, allow outbound HTTPS to your
`*.palantirfoundry.com` hostname.

## Usage

```bash
# Verify config + connectivity (no writes)
python scripts/foundry_client.py --verify

# Export locally
make foundry-export                    # CSV to stdout
python scripts/foundry_export.py --format jsonl -o runs.jsonl
python scripts/foundry_export.py --format ontology
python scripts/foundry_export.py --repo reflective-lantern

# Export + upload in one step
make foundry-sync
python scripts/foundry_sync.py --format jsonl

# Manual upload of any file
python scripts/foundry_client.py --upload runs.csv --dry-run
python scripts/foundry_client.py --upload runs.csv
```

`foundry_sync` (and the "Foundry export" step in `run_all_checks.py`) work
without credentials — they export locally and skip the upload, so CI and
sandboxed runs stay green.

## Upload semantics

Each sync opens a `SNAPSHOT` transaction, uploads three files, and commits:

| File | Contents |
|---|---|
| `reflective_lantern_runs.csv` (or `.jsonl`) | one row per automation run |
| `reflective_lantern_ontology.json` | Repository / AutomationRun ontology objects |
| `manifest.json` | row count, format, generator version |

On any failure the transaction is aborted so the dataset never sees a
partial write. Re-running replaces the snapshot; `run_key` keeps rows
stable for downstream upserts. After commit the sync lists the dataset's
files and warns if anything it uploaded is not visible. Pass
`--no-ontology` to upload only the tabular export and manifest.

## Reading data back

`FoundryClient.read_table()` wraps the `readTable` endpoint:

```bash
python scripts/foundry_client.py --read-table > current.csv
```

This requires a schema applied to the dataset (open the dataset in
Foundry and click *Apply schema* once after the first upload).
