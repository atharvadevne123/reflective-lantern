"""Initial schema: credit_applications and drift_logs tables.

Revision ID: 0001
Revises:
Create Date: 2026-11-11 00:00:00.000000
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
        "credit_applications",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("applicant_id", sa.String(), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("risk_category", sa.String(), nullable=True),
        sa.Column("default_probability", sa.Float(), nullable=True),
        sa.Column("annual_income", sa.Float(), nullable=True),
        sa.Column("debt_to_income", sa.Float(), nullable=True),
        sa.Column("num_derogatory_marks", sa.Integer(), nullable=True),
        sa.Column("num_credit_inquiries", sa.Integer(), nullable=True),
        sa.Column("credit_utilization", sa.Float(), nullable=True),
        sa.Column("num_open_accounts", sa.Integer(), nullable=True),
        sa.Column("loan_amount", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_credit_applications_applicant_id", "credit_applications", ["applicant_id"])
    op.create_index("ix_credit_applications_id", "credit_applications", ["id"])

    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("feature_name", sa.String(), nullable=True),
        sa.Column("ks_statistic", sa.Float(), nullable=True),
        sa.Column("p_value", sa.Float(), nullable=True),
        sa.Column("drift_detected", sa.Boolean(), nullable=True),
        sa.Column("logged_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drift_logs_id", "drift_logs", ["id"])


def downgrade() -> None:
    op.drop_index("ix_drift_logs_id", table_name="drift_logs")
    op.drop_table("drift_logs")
    op.drop_index("ix_credit_applications_id", table_name="credit_applications")
    op.drop_index("ix_credit_applications_applicant_id", table_name="credit_applications")
    op.drop_table("credit_applications")
