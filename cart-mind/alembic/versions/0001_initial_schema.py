"""Initial Cart-Mind schema.

Revision ID: 0001
Revises:
Create Date: 2026-08-05
"""

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the four core Cart-Mind tables."""
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("correlation_id", sa.String(64), index=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("item_id", sa.String(64)),
        sa.Column("prediction_type", sa.String(32)),
        sa.Column("score", sa.Float),
        sa.Column("model_version", sa.String(32)),
        sa.Column("latency_ms", sa.Float),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("feature_name", sa.String(128)),
        sa.Column("ks_statistic", sa.Float),
        sa.Column("p_value", sa.Float),
        sa.Column("drift_detected", sa.Integer),
        sa.Column("window_size", sa.Integer),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table(
        "user_profiles",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("user_id", sa.String(64), unique=True, index=True),
        sa.Column("segment", sa.String(32)),
        sa.Column("avg_order_value", sa.Float),
        sa.Column("purchase_count", sa.Integer),
        sa.Column("preferred_category", sa.String(64)),
        sa.Column("created_at", sa.DateTime),
        sa.Column("updated_at", sa.DateTime),
    )
    op.create_table(
        "item_catalog",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("item_id", sa.String(64), unique=True, index=True),
        sa.Column("category", sa.String(64)),
        sa.Column("subcategory", sa.String(64)),
        sa.Column("price", sa.Float),
        sa.Column("avg_rating", sa.Float),
        sa.Column("review_count", sa.Integer),
        sa.Column("description", sa.Text),
        sa.Column("created_at", sa.DateTime),
    )


def downgrade() -> None:
    """Drop all Cart-Mind tables."""
    op.drop_table("item_catalog")
    op.drop_table("user_profiles")
    op.drop_table("drift_logs")
    op.drop_table("prediction_logs")
