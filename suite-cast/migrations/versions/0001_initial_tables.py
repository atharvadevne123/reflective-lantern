"""Initial predictions and model_metrics tables.

Revision ID: 0001
Revises:
Create Date: 2026-07-15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create predictions and model_metrics tables."""
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("request_id", sa.String(36), nullable=True, unique=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("input_features", sa.JSON(), nullable=False),
        sa.Column("demand_score", sa.Float(), nullable=False),
        sa.Column("suggested_rate", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
    )
    op.create_index("ix_predictions_request_id", "predictions", ["request_id"])
    op.create_index("ix_predictions_id", "predictions", ["id"])

    op.create_table(
        "model_metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("auc_mean", sa.Float(), nullable=False),
        sa.Column("auc_std", sa.Float(), nullable=False),
        sa.Column("n_features", sa.Integer(), nullable=False),
        sa.Column("trained_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_model_metrics_model_version", "model_metrics", ["model_version"])


def downgrade() -> None:
    """Drop predictions and model_metrics tables."""
    op.drop_index("ix_model_metrics_model_version", table_name="model_metrics")
    op.drop_table("model_metrics")
    op.drop_index("ix_predictions_id", table_name="predictions")
    op.drop_index("ix_predictions_request_id", table_name="predictions")
    op.drop_table("predictions")
