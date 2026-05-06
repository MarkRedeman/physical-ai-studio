"""Command line interface for interacting with the Physical AI Studio application."""

import asyncio
import sys
from pathlib import Path
from uuid import UUID

import click

from db.engine import get_sync_db_session
from db.migration import MigrationManager
from db.schema import (
    CalibrationValuesDB,
    DatasetDB,
    JobDB,
    ProjectCameraDB,
    ProjectDB,
    ProjectEnvironmentDB,
    ProjectRobotDB,
    SnapshotDB,
)
from services import ModelImportService
from settings import get_settings

settings = get_settings()
migration_manager = MigrationManager(settings)


@click.group()
def cli() -> None:
    """Physical AI Studio CLI"""


@cli.command()
@click.option("--target-path", default="docs/openapi.json")
def gen_api(target_path: str) -> None:
    """Generate OpenAPI specification JSON file."""
    # Importing create_openapi imports threading which is slow. Importing here to not slow down other cli commands.
    from create_openapi import create_openapi

    try:
        create_openapi(target_path=target_path)
        click.echo("✓ OpenAPI specification generated successfully!")
    except Exception as e:
        click.echo(f"✗ Failed to generate OpenAPI specification: {e}")
        sys.exit(1)
    click.echo("Waiting for threading to finish...")


@cli.command()
def init_db() -> None:
    """Initialize database with migrations"""
    click.echo("Initializing database...")

    if migration_manager.initialize_database():
        click.echo("✓ Database initialized successfully!")
        sys.exit(0)
    else:
        click.echo("✗ Database initialization failed!")
        sys.exit(1)


@cli.command()
def clean_db() -> None:
    """Remove all data from the database (clean but don't drop tables)."""
    with get_sync_db_session() as db:
        db.query(ProjectDB).delete()
        db.query(ProjectRobotDB).delete()
        db.query(CalibrationValuesDB).delete()
        db.query(ProjectCameraDB).delete()
        db.query(ProjectEnvironmentDB).delete()
        db.query(DatasetDB).delete()
        db.query(SnapshotDB).delete()
        db.query(JobDB).delete()
        db.commit()
    click.echo("✓ Database cleaned successfully!")


@cli.command()
def check_db() -> None:
    """Check database status"""
    click.echo("Checking database status...")

    # Check connection
    if not migration_manager.check_connection():
        click.echo("✗ Cannot connect to database")
        sys.exit(1)

    click.echo("✓ Database connection OK")

    # Check migration status
    needs_migration, status = migration_manager.check_migration_status()
    click.echo(f"Migration status: {status}")

    if needs_migration:
        click.echo("⚠ Database needs migration")
        sys.exit(2)
    else:
        click.echo("✓ Database is up to date")
        sys.exit(0)


@cli.command()
def migrate() -> None:
    """Run database migrations"""
    click.echo("Running database migrations...")

    if migration_manager.run_migrations():
        click.echo("✓ Migrations completed successfully!")
        sys.exit(0)
    else:
        click.echo("✗ Migration failed!")
        sys.exit(1)


@cli.command("model-import")
@click.option("--archive-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--project-id", type=click.UUID, required=True)
@click.option("--dataset-id", type=click.UUID, required=True)
@click.option("--policy", type=str, required=True)
@click.option("--model-name", type=str, required=True)
@click.option("--base-model-id", type=click.UUID, default=None)
@click.option("--version", type=int, default=1)
def model_import(
    archive_path: Path,
    project_id: UUID,
    dataset_id: UUID,
    policy: str,
    model_name: str,
    base_model_id: UUID | None,
    version: int,
) -> None:
    """Import a model archive and register it in the database."""

    async def _run_import() -> None:
        imported_model = await ModelImportService().import_model_archive(
            archive_path=archive_path,
            project_id=project_id,
            dataset_id=dataset_id,
            policy=policy,
            model_name=model_name,
            base_model_id=base_model_id,
            version=version,
        )
        click.echo(f"✓ Model imported successfully: {imported_model.id}")

    try:
        asyncio.run(_run_import())
    except Exception as e:
        click.echo(f"✗ Failed to import model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
