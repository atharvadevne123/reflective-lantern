"""Initial schema — energy_readings, prediction_log, drift_reports, retraining_log.

Revision ID: 001
Revises: None
Create Date: 2026-07-10
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "energy_readings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", sa.DateTime, nullable=True),
        sa.Column("load_mw", sa.Float, nullable=False),
        sa.Column("temperature_c", sa.Float),
        sa.Column("humidity_pct", sa.Float),
        sa.Column("hour", sa.Integer, nullable=False),
        sa.Column("day_of_week", sa.Integer, nullable=False),
        sa.Column("month", sa.Integer, nullable=False),
        sa.Column("is_weekend", sa.Boolean, default=False),
        sa.Column("region", sa.String(64), default="default"),
        sa.Column("source", sa.String(64), default="api"),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table(
        "prediction_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", sa.DateTime),
        sa.Column("predicted_load_mw", sa.Float, nullable=False),
        sa.Column("actual_load_mw", sa.Float),
        sa.Column("model_version", sa.String(32), default="1.0.0"),
        sa.Column("region", sa.String(64), default="default"),
        sa.Column("features_json", sa.Text),
        sa.Column("latency_ms", sa.Float),
        sa.Column("request_id", sa.String(64)),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", sa.DateTime),
        sa.Column("feature_name", sa.String(128), nullable=False),
        sa.Column("ks_statistic", sa.Float, nullable=False),
        sa.Column("p_value", sa.Float, nullable=False),
        sa.Column("drift_detected", sa.Boolean, default=False),
        sa.Column("reference_window", sa.Integer, default=1000),
        sa.Column("current_window", sa.Integer, default=200),
        sa.Column("model_version", sa.String(32), default="1.0.0"),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table(
        "retraining_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", sa.DateTime),
        sa.Column("trigger", sa.String(64), default="manual"),
        sa.Column("samples_used", sa.Integer, nullable=False),
        sa.Column("auc_before", sa.Float),
        sa.Column("auc_after", sa.Float),
        sa.Column("rmse_after", sa.Float),
        sa.Column("r2_after", sa.Float),
        sa.Column("status", sa.String(32), default="success"),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime),
    )


def downgrade() -> None:
    op.drop_table("retraining_log")
    op.drop_table("drift_reports")
    op.drop_table("prediction_log")
    op.drop_table("energy_readings")
