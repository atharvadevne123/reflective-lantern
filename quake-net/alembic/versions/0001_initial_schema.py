"""Initial Quake-Net schema.

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
    """Create the seismic_events, drift_logs and model_metrics tables."""
    op.create_table(
        "seismic_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("depth_km", sa.Float(), nullable=False),
        sa.Column("station_count", sa.Integer(), nullable=False),
        sa.Column("p_wave_amplitude", sa.Float(), nullable=False),
        sa.Column("s_wave_amplitude", sa.Float(), nullable=False),
        sa.Column("epicentral_distance_km", sa.Float(), nullable=False),
        sa.Column("fault_type", sa.String(length=32), nullable=False),
        sa.Column("predicted_magnitude", sa.Float(), nullable=False),
        sa.Column("aftershock_probability", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_seismic_events_id", "seismic_events", ["id"])
    op.create_index("ix_seismic_events_created_at", "seismic_events", ["created_at"])

    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("feature_name", sa.String(length=64), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("drift_detected", sa.Boolean(), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("checked_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drift_logs_id", "drift_logs", ["id"])

    op.create_table(
        "model_metrics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("rmse", sa.Float(), nullable=False),
        sa.Column("mae", sa.Float(), nullable=False),
        sa.Column("r2", sa.Float(), nullable=False),
        sa.Column("cv_r2_mean", sa.Float(), nullable=False),
        sa.Column("cv_r2_std", sa.Float(), nullable=False),
        sa.Column("n_features", sa.Integer(), nullable=False),
        sa.Column("n_samples", sa.Integer(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("trained_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_metrics_id", "model_metrics", ["id"])


def downgrade() -> None:
    """Drop all Quake-Net tables."""
    op.drop_table("model_metrics")
    op.drop_table("drift_logs")
    op.drop_table("seismic_events")
