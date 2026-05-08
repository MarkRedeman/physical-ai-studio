"""add telemetry tables

Revision ID: d2f7c8a1b901
Revises: b9f3e2a1c7d8
Create Date: 2026-04-30 11:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2f7c8a1b901"
down_revision: str | Sequence[str] | None = "b9f3e2a1c7d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "telemetry_series",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("module", sa.String(length=64), nullable=False),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=False),
        sa.Column("labels_json", sa.JSON(), nullable=False),
        sa.Column("device_key", sa.String(length=128), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "module",
            "metric_name",
            "unit",
            "labels_json",
            "device_key",
            name="uq_telemetry_series_key",
        ),
    )

    op.create_table(
        "telemetry_points",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("series_id", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["series_id"], ["telemetry_series.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_telemetry_points_series_ts", "telemetry_points", ["series_id", "ts"], unique=False)
    op.create_index("ix_telemetry_points_ts", "telemetry_points", ["ts"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_telemetry_points_ts", table_name="telemetry_points")
    op.drop_index("ix_telemetry_points_series_ts", table_name="telemetry_points")
    op.drop_table("telemetry_points")
    op.drop_table("telemetry_series")
