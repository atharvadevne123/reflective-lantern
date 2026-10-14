"""Initial schema: prediction_logs, drift_logs, model_metrics.

Revision ID: 0001
Revises:
Create Date: 2026-10-14 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("prediction", sa.Integer, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("signal", sa.String(8), nullable=False),
        sa.Column("horizon_days", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("feature_name", sa.String(64), nullable=False),
        sa.Column("ks_statistic", sa.Float, nullable=False),
        sa.Column("p_value", sa.Float, nullable=False),
        sa.Column("drift_detected", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "model_metrics",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("auc_roc", sa.Float, nullable=False),
        sa.Column("accuracy", sa.Float, nullable=False),
        sa.Column("n_samples", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("model_metrics")
    op.drop_table("drift_logs")
    op.drop_table("prediction_logs")
