"""Model management CLI commands."""

import asyncio
import sys
from pathlib import Path
from uuid import UUID

import click


@click.group()
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


@models.command("export")
@click.argument("model_id", type=click.UUID)
@click.option(
    "--backend",
    "backends",
    multiple=True,
    type=click.Choice(["torch", "openvino", "onnx", "executorch"]),
    help="Backend(s) to export to (repeatable). Defaults to torch and openvino.",
)
@click.option(
    "--compress/--no-compress",
    default=True,
    help="Apply NNCF INT8 weight compression to the OpenVINO export (default: compress)",
)
def export_model(model_id: UUID, backends: tuple[str, ...], compress: bool) -> None:
    """Add requested backend exports to an existing trained model."""
    from services.model_export_service import ModelExportError, ModelExportService

    selected = list(backends) or ["torch", "openvino"]
    click.echo(f"Re-exporting model {model_id} to: {', '.join(selected)}")
    if compress and "openvino" in selected:
        click.echo("  NNCF INT8 weight compression enabled")

    async def _run_export() -> None:
        exported = await ModelExportService.export_model(
            model_id=model_id,
            backends=selected,
            compress=compress,
        )
        click.echo("Model exports added successfully!")
        click.echo(f"Model ID: {exported.id}")
        click.echo(f"Model path: {exported.path}")

    try:
        asyncio.run(_run_export())
    except ModelExportError as e:
        click.echo(f"Export failed: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}")
        sys.exit(1)


@models.command("compress")
@click.argument("model_id", type=click.UUID)
@click.option("--name", default=None, type=str, help="Name for the compressed model (default: '<original> (INT8)')")
def compress_weights(model_id: UUID, name: str | None) -> None:
    """Compress an exported OpenVINO model's weights to INT8 using NNCF.

    Creates a new model record with compressed weights, linked to the original
    via parent_model_id. Only the OpenVINO export directory is included.
    """
    from services.model_compression_service import ModelCompressionError, ModelCompressionService

    click.echo(f"Compressing model {model_id} with NNCF INT8_SYM...")

    async def _run_compression() -> None:
        compressed = await ModelCompressionService.compress_model(model_id=model_id, name=name)
        click.echo("Model compressed successfully!")
        click.echo(f"Compressed model ID: {compressed.id}")
        click.echo(f"Compressed model path: {compressed.path}")

    try:
        asyncio.run(_run_compression())
    except ModelCompressionError as e:
        click.echo(f"Compression failed: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}")
        sys.exit(1)
