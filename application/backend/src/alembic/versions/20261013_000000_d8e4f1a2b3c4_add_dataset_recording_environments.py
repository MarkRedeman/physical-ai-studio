"""Allow datasets without an assigned recording environment.

Revision ID: d8e4f1a2b3c4
Revises: 7c1a9e2b4d6f
Create Date: 2026-10-13 00:00:00.000000
"""

from collections.abc import Sequence
from uuid import uuid4

import sqlalchemy as sa

from alembic import op

revision: str = "d8e4f1a2b3c4"
down_revision: str | Sequence[str] | None = "7c1a9e2b4d6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Move legacy dataset environment assignments into recording associations."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if inspector.has_table("dataset_recording_environments"):
        # SQLite executes DDL outside transactions. A prior interrupted upgrade
        # can therefore leave this table behind without recording the Alembic
        # revision. Resume safely from that partially-applied state.
        recording_environments = sa.Table("dataset_recording_environments", sa.MetaData(), autoload_with=connection)
    else:
        recording_environments = op.create_table(
            "dataset_recording_environments",
            sa.Column("id", sa.Text(), nullable=False),
            sa.Column("dataset_id", sa.Text(), nullable=False),
            sa.Column("environment_id", sa.Text(), nullable=False),
            sa.Column("mappings", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["environment_id"], ["project_environments.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "dataset_id",
                "environment_id",
                name="uq_dataset_recording_environments_dataset_environment",
            ),
        )

    dataset_columns = {column["name"] for column in inspector.get_columns("datasets")}
    if "environment_id" not in dataset_columns:
        return

    legacy_assignments = connection.execute(sa.text("SELECT id, environment_id FROM datasets")).mappings().all()
    existing_associations = {
        (row["dataset_id"], row["environment_id"])
        for row in connection.execute(
            sa.text("SELECT dataset_id, environment_id FROM dataset_recording_environments")
        ).mappings()
    }
    association_rows = [
        {"id": str(uuid4()), "dataset_id": assignment["id"], "environment_id": assignment["environment_id"]}
        for assignment in legacy_assignments
        if (assignment["id"], assignment["environment_id"]) not in existing_associations
    ]
    if association_rows:
        connection.execute(recording_environments.insert(), association_rows)

    # SQLite implements DROP COLUMN by rebuilding the table. The association
    # table already references datasets, so temporarily disable FK checks while
    # Alembic replaces the legacy table (matching the earlier dataset migration
    # strategy in aa0f562acb23).
    if connection.dialect.name == "sqlite":
        connection.execute(sa.text("PRAGMA foreign_keys = OFF"))
    try:
        with op.batch_alter_table("datasets") as batch_op:
            batch_op.drop_column("environment_id")
    finally:
        if connection.dialect.name == "sqlite":
            connection.execute(sa.text("PRAGMA foreign_keys = ON"))


def downgrade() -> None:
    """Restore the legacy environment column from each dataset's first association."""
    with op.batch_alter_table("datasets") as batch_op:
        batch_op.add_column(sa.Column("environment_id", sa.Text(), nullable=True))

    op.execute(
        sa.text(
            "UPDATE datasets SET environment_id = ("
            "SELECT environment_id FROM dataset_recording_environments "
            "WHERE dataset_recording_environments.dataset_id = datasets.id "
            "ORDER BY created_at, id LIMIT 1)"
        )
    )
    op.drop_table("dataset_recording_environments")

    # Datasets created without a recording environment cannot be represented by
    # the old schema, so the restored column intentionally remains nullable.
