"""Initial schema: incidents, predictions, drift_alerts.

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
    """Create the three core tables and their indexes."""
    op.create_table(
        "incidents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("service_name", sa.String(length=128), nullable=False),
        sa.Column("cpu_usage_pct", sa.Float(), nullable=False),
        sa.Column("memory_usage_pct", sa.Float(), nullable=False),
        sa.Column("error_rate_per_min", sa.Float(), nullable=False),
        sa.Column("latency_p99_ms", sa.Float(), nullable=False),
        sa.Column("request_rate_per_sec", sa.Float(), nullable=False),
        sa.Column("disk_io_util_pct", sa.Float(), nullable=False),
        sa.Column("is_incident", sa.Boolean(), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_incidents_service_name", "incidents", ["service_name"])
    op.create_index("ix_incidents_created_at", "incidents", ["created_at"])
    op.create_index(
        "ix_incidents_service_created", "incidents", ["service_name", "created_at"]
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("incident_id", sa.Integer(), nullable=True),
        sa.Column("service_name", sa.String(length=128), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("predicted_incident", sa.Boolean(), nullable=False),
        sa.Column("predicted_severity", sa.String(length=16), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_incident_id", "predictions", ["incident_id"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index(
        "ix_predictions_incident_flag", "predictions", ["predicted_incident"]
    )

    op.create_table(
        "drift_alerts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("feature_name", sa.String(length=64), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("drifted", sa.Boolean(), nullable=False),
        sa.Column("reference_window", sa.String(length=32), nullable=False),
        sa.Column("current_window", sa.String(length=32), nullable=False),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drift_alerts_created_at", "drift_alerts", ["created_at"])
    op.create_index(
        "ix_drift_alerts_drifted_created",
        "drift_alerts",
        ["drifted", "created_at"],
    )


def downgrade() -> None:
    """Drop the three core tables."""
    op.drop_table("drift_alerts")
    op.drop_table("predictions")
    op.drop_table("incidents")
