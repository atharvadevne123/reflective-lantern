"""Initial schema.

Revision ID: 001
Revises:
Create Date: 2026-07-06
"""

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "properties",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("address", sa.String(500)),
        sa.Column("zipcode", sa.String(10), index=True),
        sa.Column("city", sa.String(100)),
        sa.Column("state", sa.String(50)),
        sa.Column("sqft", sa.Float),
        sa.Column("bedrooms", sa.Integer),
        sa.Column("bathrooms", sa.Float),
        sa.Column("lot_size", sa.Float),
        sa.Column("year_built", sa.Integer),
        sa.Column("renovation_year", sa.Integer, nullable=True),
        sa.Column("condition_score", sa.Float),
        sa.Column("list_price", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("property_id", sa.Integer, nullable=True),
        sa.Column("predicted_value", sa.Float),
        sa.Column("investment_score", sa.Float, nullable=True),
        sa.Column("model_version", sa.String(50)),
        sa.Column("features_json", sa.Text),
        sa.Column("correlation_id", sa.String(50), index=True),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("feature_name", sa.String(100)),
        sa.Column("ks_statistic", sa.Float),
        sa.Column("p_value", sa.Float),
        sa.Column("drift_detected", sa.Boolean),
        sa.Column("sample_size", sa.Integer),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table(
        "neighborhood_stats",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("zipcode", sa.String(10), unique=True, index=True),
        sa.Column("median_price", sa.Float),
        sa.Column("median_price_per_sqft", sa.Float),
        sa.Column("school_score", sa.Float),
        sa.Column("transit_score", sa.Float),
        sa.Column("walkability_score", sa.Float),
        sa.Column("crime_rate", sa.Float),
        sa.Column("avg_rental_yield", sa.Float),
        sa.Column("updated_at", sa.DateTime),
    )


def downgrade() -> None:
    op.drop_table("neighborhood_stats")
    op.drop_table("drift_reports")
    op.drop_table("prediction_logs")
    op.drop_table("properties")
