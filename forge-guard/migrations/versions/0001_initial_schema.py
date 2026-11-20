"""Initial schema — prediction_logs, drift_reports, retraining_runs.

Revision ID: 0001
Revises:
Create Date: 2026-06-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("timestamp", sa.DateTime, nullable=False),
        sa.Column("temperature", sa.Float, nullable=False),
        sa.Column("pressure", sa.Float, nullable=False),
        sa.Column("vibration", sa.Float, nullable=False),
        sa.Column("cycle_time", sa.Float, nullable=False),
        sa.Column("tool_wear", sa.Float, nullable=False),
        sa.Column("power_consumption", sa.Float, nullable=False),
        sa.Column("humidity", sa.Float, nullable=False),
        sa.Column("prediction", sa.Integer, nullable=False),
        sa.Column("defect_probability", sa.Float, nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("correlation_id", sa.String(64), nullable=True),
    )
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("timestamp", sa.DateTime, nullable=False),
        sa.Column("feature", sa.String(64), nullable=False),
        sa.Column("ks_statistic", sa.Float, nullable=False),
        sa.Column("p_value", sa.Float, nullable=False),
        sa.Column("drift_detected", sa.Boolean, nullable=False),
        sa.Column("window_size", sa.Integer, nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
    )
    op.create_table(
        "retraining_runs",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("started_at", sa.DateTime, nullable=False),
        sa.Column("finished_at", sa.DateTime, nullable=True),
        sa.Column("trigger", sa.String(32), nullable=False),
        sa.Column("auc_before", sa.Float, nullable=True),
        sa.Column("auc_after", sa.Float, nullable=True),
        sa.Column("n_samples", sa.Integer, nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("retraining_runs")
    op.drop_table("drift_reports")
    op.drop_table("prediction_logs")
