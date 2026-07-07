"""Initial schema: network_events, predictions, drift_logs.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-06-29
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables."""
    op.create_table(
        "network_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("src_ip", sa.String(45), nullable=True),
        sa.Column("dst_ip", sa.String(45), nullable=True),
        sa.Column("protocol", sa.String(16), nullable=True),
        sa.Column("payload_bytes", sa.Integer(), nullable=False, default=0),
        sa.Column("duration_ms", sa.Float(), nullable=False, default=0.0),
        sa.Column("flags", sa.String(64), nullable=True),
        sa.Column("raw_features", sa.Text(), nullable=True),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("attack_type", sa.String(64), nullable=True),
        sa.Column("model_version", sa.String(32), nullable=True),
        sa.Column("drift_score", sa.Float(), nullable=True),
    )

    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("recorded_at", sa.DateTime(), nullable=False),
        sa.Column("feature_name", sa.String(64), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("drift_detected", sa.Integer(), nullable=False, default=0),
    )

    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("ix_drift_logs_recorded_at", "drift_logs", ["recorded_at"])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("drift_logs")
    op.drop_table("predictions")
    op.drop_table("network_events")
