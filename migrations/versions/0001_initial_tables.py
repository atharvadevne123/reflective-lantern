"""Initial tables: prediction_logs and drift_logs.

Revision ID: 0001
Revises:
Create Date: 2026-07-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("hour", sa.Integer(), nullable=False),
        sa.Column("day_of_week", sa.Integer(), nullable=False),
        sa.Column("route_id", sa.String(64), nullable=False),
        sa.Column("road_type", sa.String(32), nullable=False),
        sa.Column("congestion_level", sa.Integer(), nullable=False),
        sa.Column("congestion_prob", sa.Float(), nullable=False),
        sa.Column("incident_score", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("features_json", sa.Text(), nullable=True),
    )
    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("feature_name", sa.String(64), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("drift_detected", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("drift_logs")
    op.drop_table("prediction_logs")
