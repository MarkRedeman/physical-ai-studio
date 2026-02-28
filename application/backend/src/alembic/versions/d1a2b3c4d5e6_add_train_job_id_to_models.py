"""Add train_job_id to models table

Revision ID: d1a2b3c4d5e6
Revises: c3e4e04cc003
Create Date: 2026-02-28 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "c3e4e04cc003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add train_job_id foreign key column to the models table."""
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.add_column(sa.Column("train_job_id", sa.Text(), nullable=True))
        batch_op.create_foreign_key(
            "fk_models_train_job_id_jobs",
            "jobs",
            ["train_job_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    """Remove train_job_id column from the models table."""
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.drop_constraint("fk_models_train_job_id_jobs", type_="foreignkey")
        batch_op.drop_column("train_job_id")
