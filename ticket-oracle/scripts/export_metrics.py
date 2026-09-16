"""Export Ticket-Oracle prediction metrics to CSV for analysis.

Reads the PredictionLog table and writes a timestamped CSV file
to the current directory for use in dashboards or ad-hoc analysis.

Usage:
    python scripts/export_metrics.py [--out predictions_export.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import PredictionLog, init_db, _get_session_factory


FIELDS = [
    "id",
    "ticket_id",
    "predicted_priority",
    "priority_confidence",
    "sla_breach_risk",
    "sla_breached",
    "resolution_hours_estimate",
    "department",
    "channel",
    "created_at",
]


def export(out_path: str) -> None:
    init_db()
    Session = _get_session_factory()

    with Session() as session:
        rows = session.query(PredictionLog).order_by(PredictionLog.created_at).all()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field, "") for field in FIELDS})

    print(f"Exported {len(rows)} rows to {out_path}")


def main() -> None:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Export Ticket-Oracle prediction log to CSV")
    parser.add_argument("--out", default=f"predictions_{ts}.csv", help="Output CSV path")
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
