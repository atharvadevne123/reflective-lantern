"""Initial schema: sensor_readings, anomaly_events, predictions, drift_logs

Revision ID: 0001
Revises:
Create Date: 2026-07-08

"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sensor_readings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("sensor_id", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("values", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_sensor_readings_sensor_id", "sensor_readings", ["sensor_id"])
    op.create_index("ix_sensor_readings_timestamp", "sensor_readings", ["timestamp"])
    op.create_index("ix_sensor_timestamp", "sensor_readings", ["sensor_id", "timestamp"])

    op.create_table(
        "anomaly_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("sensor_id", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=False),
        sa.Column("root_cause", sa.Text()),
        sa.Column("features_snapshot", sa.JSON()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_anomaly_events_sensor_id", "anomaly_events", ["sensor_id"])
    op.create_index("ix_anomaly_sensor_ts", "anomaly_events", ["sensor_id", "timestamp"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("sensor_id", sa.String(64), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("predicted_values", sa.JSON(), nullable=False),
        sa.Column("confidence_interval", sa.JSON()),
        sa.Column("model_version", sa.String(32)),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_predictions_sensor_id", "predictions", ["sensor_id"])

    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("sensor_id", sa.String(64), nullable=False),
        sa.Column("feature_name", sa.String(128), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("drift_detected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_drift_logs_sensor_id", "drift_logs", ["sensor_id"])


def downgrade() -> None:
    op.drop_table("drift_logs")
    op.drop_table("predictions")
    op.drop_table("anomaly_events")
    op.drop_table("sensor_readings")
