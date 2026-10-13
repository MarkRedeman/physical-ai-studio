"""Add a default recording environment per dataset.

Revision ID: e5f6a7b8c9d0
Revises: d8e4f1a2b3c4
Create Date: 2026-10-14 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | Sequence[str] | None = "d8e4f1a2b3c4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add and backfill the default association."""
    op.add_column(
        "dataset_recording_environments",
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.execute(
        sa.text(
            "UPDATE dataset_recording_environments SET is_default = 1 "
            "WHERE id IN (SELECT MIN(id) FROM dataset_recording_environments GROUP BY dataset_id)"
        )
    )
    op.create_index(
        "uq_dataset_recording_environments_default",
        "dataset_recording_environments",
        ["dataset_id"],
        unique=True,
        sqlite_where=sa.text("is_default = 1"),
        postgresql_where=sa.text("is_default"),
    )


def downgrade() -> None:
    """Remove the default association marker."""
    op.drop_index("uq_dataset_recording_environments_default", table_name="dataset_recording_environments")
    op.drop_column("dataset_recording_environments", "is_default")
