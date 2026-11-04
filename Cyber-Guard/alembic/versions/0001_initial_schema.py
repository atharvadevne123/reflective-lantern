"""Initial schema: prediction_logs and drift_logs.

Revision ID: 0001
Revises:
Create Date: 2026-09-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the prediction and drift log tables."""
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("src_bytes", sa.Float(), nullable=False),
        sa.Column("dst_bytes", sa.Float(), nullable=False),
        sa.Column("duration", sa.Float(), nullable=False),
        sa.Column("protocol_type", sa.String(length=16), nullable=False),
        sa.Column("service", sa.String(length=32), nullable=False),
        sa.Column("flag", sa.String(length=16), nullable=False),
        sa.Column("prediction", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    # Drift queries filter on timestamp ranges over the full table, so this
    # index is what keeps /api/v1/drift from degrading into a scan.
    op.create_index("ix_prediction_logs_timestamp", "prediction_logs", ["timestamp"])
    op.create_index("ix_prediction_logs_id", "prediction_logs", ["id"])

    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("feature", sa.String(length=64), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("drift_detected", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drift_logs_timestamp", "drift_logs", ["timestamp"])
    op.create_index("ix_drift_logs_id", "drift_logs", ["id"])


def downgrade() -> None:
    """Drop the prediction and drift log tables."""
    op.drop_index("ix_drift_logs_id", table_name="drift_logs")
    op.drop_index("ix_drift_logs_timestamp", table_name="drift_logs")
    op.drop_table("drift_logs")
    op.drop_index("ix_prediction_logs_id", table_name="prediction_logs")
    op.drop_index("ix_prediction_logs_timestamp", table_name="prediction_logs")
    op.drop_table("prediction_logs")
