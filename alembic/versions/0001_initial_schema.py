"""Initial schema: predictions and drift_logs tables.

Revision ID: 0001
Revises:
Create Date: 2026-08-20
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
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("created_at", sa.DateTime(), index=True),
        sa.Column("carrier", sa.String(length=64)),
        sa.Column("distance_km", sa.Float()),
        sa.Column("weight_kg", sa.Float()),
        sa.Column("route_type", sa.String(length=32)),
        sa.Column("hour_of_day", sa.Integer()),
        sa.Column("day_of_week", sa.Integer()),
        sa.Column("predicted_minutes", sa.Float()),
        sa.Column("confidence", sa.Float()),
        sa.Column("model_version", sa.String(length=32)),
    )
    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("feature", sa.String(length=64)),
        sa.Column("ks_statistic", sa.Float()),
        sa.Column("p_value", sa.Float()),
        sa.Column("drift_detected", sa.Integer()),
    )


def downgrade() -> None:
    op.drop_table("drift_logs")
    op.drop_table("predictions")
