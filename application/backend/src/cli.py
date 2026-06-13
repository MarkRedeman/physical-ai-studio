"""Command line interface for interacting with the Physical AI Studio application."""

import asyncio
import json
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
from settings import get_settings
from storage_migration import StorageMigrationError, migrate_default_storage_dir

settings = get_settings()


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

    migration_manager = MigrationManager(settings)
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

    migration_manager = MigrationManager(settings)

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

    try:
        migrate_default_storage_dir(settings)
    except StorageMigrationError as e:
        click.echo(f"✗ Storage migration failed: {e}", err=True)
        sys.exit(1)

    migration_manager = MigrationManager(settings)
    if migration_manager.run_migrations():
        click.echo("✓ Migrations completed successfully!")
        sys.exit(0)
    else:
        click.echo("✗ Migration failed!")
        sys.exit(1)


# =============================================================================
# Models subcommand group
# =============================================================================


@cli.group()
def models() -> None:
    """Model management commands."""


@models.command("import-dir")
@click.option("--source-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--project-id", required=True, type=click.UUID)
@click.option("--dataset-id", required=True, type=click.UUID)
@click.option("--model-name", required=True, type=str)
@click.option("--move/--copy", default=False, show_default=True)
@click.option("--base-model-id", type=click.UUID, default=None)
@click.option("--version", type=int, default=1, show_default=True)
def import_dir(
    source_dir: Path,
    project_id: UUID,
    dataset_id: UUID,
    model_name: str,
    move: bool,
    base_model_id: UUID | None,
    version: int,
) -> None:
    """Import a model from an existing folder (copy or move) trained by Physical AI Studio.

    The dataset_id should reference a dataset that uses the same environment as the
    original training. The inference UI uses this dataset to determine which environment
    to load when running the model.
    """
    from services.model_import_service import ModelImportService

    click.echo(f"Importing model from folder: {source_dir}")
    click.echo(f"Mode: {'move' if move else 'copy'}")

    async def _run_import() -> None:
        service = ModelImportService()
        model = await service.import_model_directory(
            source_dir=source_dir,
            project_id=project_id,
            dataset_id=dataset_id,
            model_name=model_name,
            move=move,
            base_model_id=base_model_id,
            version=version,
        )
        click.echo("Model imported successfully!")
        click.echo(f"Model ID: {model.id}")
        click.echo(f"Model path: {model.path}")

    try:
        asyncio.run(_run_import())
    except Exception as e:
        click.echo(f"Model import failed: {e}")
        sys.exit(1)


@models.command("import-from-lerobot")
@click.option("--source-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--project-id", required=True, type=click.UUID)
@click.option("--dataset-id", required=True, type=click.UUID)
@click.option("--model-name", required=True, type=str)
@click.option("--move/--copy", default=False, show_default=True)
@click.option("--base-model-id", type=click.UUID, default=None)
@click.option("--version", type=int, default=1, show_default=True)
@click.option("--allow-unknown-policy", is_flag=True, default=False)
def import_from_lerobot(
    source_dir: Path,
    project_id: UUID,
    dataset_id: UUID,
    model_name: str,
    move: bool,
    base_model_id: UUID | None,
    version: int,
    allow_unknown_policy: bool,
) -> None:
    """Import a LeRobot directory and convert it into a PAS model folder."""
    from services.model_import_service import ModelImportService

    click.echo(f"Importing LeRobot model from folder: {source_dir}")
    click.echo(f"Mode: {'move' if move else 'copy'}")

    async def _run_import() -> None:
        service = ModelImportService()
        model = await service.import_from_lerobot_directory(
            source_dir=source_dir,
            project_id=project_id,
            dataset_id=dataset_id,
            model_name=model_name,
            move=move,
            base_model_id=base_model_id,
            version=version,
            allow_unknown_policy=allow_unknown_policy,
        )
        click.echo("LeRobot model imported successfully!")
        click.echo(f"Model ID: {model.id}")
        click.echo(f"Model path: {model.path}")

    try:
        asyncio.run(_run_import())
    except Exception as e:
        click.echo(f"Model import failed: {e}")
        sys.exit(1)



@models.command("lerobot")
@click.option("--model-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def lerobot(model_dir: Path) -> None:
    """Re-export a LeRobot pretrained directory into supported backends.

    Example:
    uv run src/cli.py models lerobot --model-dir /path/to/model/exports/lerobot
    """
    from loguru import logger

    from physicalai.export import ExportablePolicyMixin
    from physicalai.policies import ACT, Pi05, SmolVLA
    from physicalai.policies.base import Policy
    from physicalai.policies.lerobot import LeRobotPolicy

    def load_policy_from_lerobot_dir(path: Path) -> Policy:
        config_path = path / "config.json"
        with config_path.open(encoding="utf-8") as f:
            policy_type = str(json.load(f).get("type", "")).lower()

        native_loaders: dict[str, type[Policy]] = {
            "act": ACT,
            "pi05": Pi05,
            "smolvla": SmolVLA,
        }

        native_cls = native_loaders.get(policy_type)
        if native_cls is None:
            logger.info("Policy type '{}' does not have a native converter path, using LeRobot wrapper", policy_type)
            return LeRobotPolicy.from_pretrained(path)

        model_root = path.parent.parent
        checkpoint_candidates = [
            model_root / "model.ckpt",
            model_root / "last.ckpt",
            path / "model.ckpt",
            path / "last.ckpt",
        ]

        for checkpoint_path in checkpoint_candidates:
            if checkpoint_path.exists():
                logger.info("Loading native '{}' checkpoint from {}", policy_type, checkpoint_path)
                return native_cls.load_from_checkpoint(str(checkpoint_path))

        if policy_type == "smolvla":
            logger.info("Loading native '{}' from LeRobot pretrained directory", policy_type)
            return SmolVLA(pretrained_name_or_path=path)
        if policy_type == "pi05":
            logger.info("Loading native '{}' from LeRobot pretrained directory", policy_type)
            return Pi05(pretrained_name_or_path=path)
        if policy_type == "act":
            act_checkpoint_candidates = [
                path / "model.pt",
                path / "model.ckpt",
            ]
            for checkpoint_path in act_checkpoint_candidates:
                if checkpoint_path.exists():
                    logger.info("Loading native '{}' from {}", policy_type, checkpoint_path)
                    try:
                        return ACT.load_from_checkpoint(str(checkpoint_path), weights_only=False)
                    except Exception as e:
                        logger.warning("Failed loading ACT checkpoint from {}: {}", checkpoint_path, e)
            if (path / "model.safetensors").exists():
                from services.model_import_service import ModelImportService

                logger.info("Loading native '{}' from LeRobot safetensors directory", policy_type)
                return ModelImportService()._load_native_act_from_lerobot_safetensors(path)  # noqa: SLF001

        logger.warning(
            "Could not find a native Lightning checkpoint for '{}' near {}. "
            "Falling back to LeRobot wrapper (likely torch-only export support).",
            policy_type,
            path,
        )
        return LeRobotPolicy.from_pretrained(path)

    click.echo(f"Re-export model from LeRobot folder: {model_dir}")
    policy = load_policy_from_lerobot_dir(model_dir)


    async def _run_export() -> None:
        if not isinstance(policy, ExportablePolicyMixin):
            logger.info("Skipping export: policy does not support export backends")
            return

        logger.info("Starting model export for trained policy")
        for backend in policy.get_supported_export_backends():
            backend_name = backend.value if hasattr(backend, "value") else str(backend)
            try:
                logger.info("Exporting model to {} format", backend_name)
                export_dir = model_dir / "exports" / backend_name

                if export_dir.exists():
                    logger.info("Skipping: {}", backend_name)
                    continue

                policy.export(export_dir, backend=backend)
                logger.info("Model export to {} completed", backend_name)
            except Exception as e:
                logger.error("Failed exporting model to {} format", backend_name)
                logger.exception(e)

    try:
        asyncio.run(_run_export())
    except Exception as e:
        click.echo(f"Model import failed: {e}")
        sys.exit(1)

@models.command("reexport")
@click.option("--model-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--model-type", required=True, type=str)
def reexport(
    model_dir: Path,
    model_type: str,
) -> None:
    """Rexport

uv run src/cli.py models reexport --model-dir /home/mark/data/models/smolvla-tests/dice-cleanup-combined-smolvla-pas --model-type smolvla
uv run src/cli.py models reexport --model-dir /home/mark/data/models/smolvla-tests/dice-cleanup-combined-act-pas --model-type act

    """
    from physicalai.policies import ACT, Pi05, SmolVLA
    from physicalai.policies.base import Policy
    from physicalai.export import ExportablePolicyMixin
    from loguru import logger

    def load_policy(model_dir: Path, model_type) -> Policy:
        """Load existing model."""
        model_path = str(model_dir / "model.ckpt")

        if model_type == "act":
            return ACT.load_from_checkpoint(model_path)
        elif model_type == "pi05":
            return Pi05.load_from_checkpoint(model_path)
        elif model_type == "smolvla":
            return SmolVLA.load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Policy {model_type} not implemented.")

    click.echo(f"Re-export model from folder: {model_dir}")

    policy = load_policy(model_dir, model_type)

    async def _run_export() -> None:
        if not isinstance(policy, ExportablePolicyMixin):
            logger.info("Skipping export: policy does not support export backends")
            return

        logger.info("Starting model export for trained policy")
        for backend in policy.get_supported_export_backends():
            backend_name = backend.value if hasattr(backend, "value") else str(backend)
            try:
                logger.info("Exporting model to {} format", backend_name)
                export_dir = model_dir / "exports" / backend

                if export_dir.exists():
                    logger.info("Skipping: {}", backend_name)
                    continue

                policy.export(export_dir, backend=backend)
                logger.info("Model export to {} completed", backend_name)
            except Exception as e:
                logger.error("Failed exporting model to {} format", backend_name)
                logger.exception(e)

    try:
        asyncio.run(_run_export())
    except Exception as e:
        click.echo(f"Model import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli()

