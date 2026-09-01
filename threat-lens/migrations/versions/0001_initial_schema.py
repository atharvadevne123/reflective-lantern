"""Initial Threat-Lens schema.

Revision ID: 0001
Revises:
Create Date: 2026-08-26
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("correlation_id", sa.String(64)),
        sa.Column("src_bytes", sa.Float()),
        sa.Column("dst_bytes", sa.Float()),
        sa.Column("duration", sa.Float()),
        sa.Column("protocol_type", sa.String(16)),
        sa.Column("service", sa.String(32)),
        sa.Column("flag", sa.String(16)),
        sa.Column("predicted_class", sa.String(32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("is_attack", sa.Integer(), nullable=False),
        sa.Column("raw_features", sa.Text()),
    )
    op.create_index("ix_prediction_logs_id", "prediction_logs", ["id"])
    op.create_index("ix_prediction_logs_correlation_id", "prediction_logs", ["correlation_id"])

    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("feature_name", sa.String(64), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("drift_detected", sa.Integer(), nullable=False),
        sa.Column("reference_n", sa.Integer()),
        sa.Column("current_n", sa.Integer()),
    )
    op.create_index("ix_drift_reports_id", "drift_reports", ["id"])

    op.create_table(
        "retraining_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("trigger_reason", sa.String(128), nullable=False),
        sa.Column("auc_before", sa.Float()),
        sa.Column("auc_after", sa.Float()),
        sa.Column("n_samples", sa.Integer()),
        sa.Column("success", sa.Integer(), nullable=False),
    )
    op.create_index("ix_retraining_events_id", "retraining_events", ["id"])


def downgrade() -> None:
    op.drop_table("retraining_events")
    op.drop_table("drift_reports")
    op.drop_table("prediction_logs")
