"""Model management CLI commands."""

import asyncio
import sys
from pathlib import Path
from uuid import UUID

import click


@click.group()
def models() -> None:
    """Import Studio models from local folders.

    For Hugging Face repositories, use hf model instead. These commands register
    models in the configured local Studio database, not a remote Studio server.
    """


@models.command("import-dir")
@click.option(
    "--source-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Existing Studio model directory or exports-only directory.",
)
@click.option("--project-id", required=True, type=click.UUID, help="Existing target Studio project UUID.")
@click.option("--dataset-id", required=True, type=click.UUID, help="Local dataset UUID in the target project.")
@click.option("--model-name", required=True, type=str, help="Display name for the imported model.")
@click.option("--move/--copy", default=False, show_default=True, help="Move removes the source; copy keeps it intact.")
@click.option("--base-model-id", type=click.UUID, default=None, help="Optional local parent model UUID for lineage.")
@click.option("--version", type=int, default=1, show_default=True, help="Version assigned to the imported model.")
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

    Copies files by default; --move removes the source directory on success.
    This command has no --dry-run, --json, or resumable-transfer interface.

    \b
    Example (replace the UUID placeholders):
      physicalai-studio models import-dir --source-dir ./model \\
        --project-id PROJECT_UUID --dataset-id DATASET_UUID --model-name "My model"
    """
    from db import get_async_db_session_ctx
    from schemas import Model, TrainJob
    from schemas.dataset import Dataset
    from services.dataset_service import DatasetService
    from services.job_service import JobService
    from services.model_import_service import ModelImportService
    from services.model_service import ModelService

    click.echo(f"Importing model from folder: {source_dir}")
    click.echo(f"Mode: {'move' if move else 'copy'}")

    async def _run_import() -> None:
        async def get_dataset(dataset_id: UUID) -> Dataset:
            async with get_async_db_session_ctx() as session:
                return await DatasetService(session).get_dataset_by_id(dataset_id)

        async def persist_import(job: TrainJob, model: Model) -> Model:
            async with get_async_db_session_ctx() as session:
                saved_job = await JobService(session).create_job(job)
                return await ModelService(session).create_model(model.model_copy(update={"train_job_id": saved_job.id}))

        service = ModelImportService(get_dataset=get_dataset, persist_import=persist_import)
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
