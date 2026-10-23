"""Model management CLI commands."""

import asyncio
import shutil
import sys
from pathlib import Path
from uuid import UUID, uuid4

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
    type=click.Choice(("torch", "openvino", "onnx", "executorch"), case_sensitive=False),
    default=("torch", "openvino"),
    show_default=True,
    help="Export backend to produce. Specify the option more than once to select multiple backends.",
)
def export(model_id: UUID, backends: tuple[str, ...]) -> None:
    """Export a persisted model checkpoint to one or more deployment backends.

    By default this produces PyTorch and OpenVINO artifacts under the model's
    ``exports/`` directory. Existing artifacts are handled by the policy export
    implementation in the same way as exports performed after training.
    """
    from db import get_async_db_session_ctx
    from schemas import Model
    from services.model_service import ModelService
    from training.job import TrainingJobSpec, _export, _load_policy_from_checkpoint, resolve_checkpoint

    async def get_model() -> Model:
        async with get_async_db_session_ctx() as session:
            return await ModelService(session).get_model_by_id(model_id)

    def report(_progress: int, message: str | None, _extra_info: dict) -> None:
        if message is not None:
            click.echo(message)

    try:
        model = asyncio.run(get_model())
        model_dir = Path(model.path)
        checkpoint = resolve_checkpoint(model_dir)
        if not checkpoint.is_file():
            raise click.ClickException(f"Model checkpoint not found: {checkpoint}")

        requested = [backend.lower() for backend in backends]
        click.echo(f"Loading model {model.id} from {checkpoint}")
        policy = _load_policy_from_checkpoint(TrainingJobSpec(policy=model.policy), checkpoint)
        _export(policy, model_dir, report, requested=requested)
        click.echo(f"Export finished for model {model.id}")
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Model export failed: {exc}") from exc


@models.command("import-websummit")
@click.option("--source-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--project-id", required=True, type=click.UUID)
@click.option("--dataset-id", required=True, type=click.UUID)
@click.option(
    "--policy",
    default="smolvla",
    show_default=True,
    type=click.Choice(("act", "smolvla", "pi05", "rldx1", "molmoact2", "xr0"), case_sensitive=False),
    help="Policy implemented by the WebSummit checkpoints.",
)
def import_websummit(  # noqa: PLR0915
    source_dir: Path, project_id: UUID, dataset_id: UUID, policy: str
) -> None:
    """Import and export every checkpoint from a WebSummit checkpoint bundle.

    SOURCE_DIR must contain ``websummit-only-no-aug`` and
    ``websummit-computex-no-aug`` directories. Each ``*.ckpt`` file directly
    inside those directories becomes a separate Studio model. Checkpoints are
    moved into Studio storage as ``model.ckpt`` and exported to Torch and
    OpenVINO.
    """
    from db import get_async_db_session_ctx
    from schemas import Model, TrainJob
    from schemas.base_job import JobStatus
    from schemas.dataset import Dataset
    from schemas.job import LocalTrainJobPayload
    from schemas.model import SNAPFLOW_PROPERTY
    from services.dataset_service import DatasetService
    from services.job_service import JobService
    from services.model_service import ModelService
    from settings import get_settings
    from training.job import TrainingJobSpec, _export, _load_policy_from_checkpoint

    policy = policy.lower()
    variants = ("websummit-only-no-aug", "websummit-computex-no-aug")
    checkpoint_groups: list[tuple[str, Path]] = []
    for variant in variants:
        variant_dir = source_dir / variant
        if not variant_dir.is_dir():
            raise click.ClickException(f"Missing required checkpoint directory: {variant_dir}")
        checkpoints = sorted(path for path in variant_dir.iterdir() if path.is_file() and path.suffix == ".ckpt")
        if not checkpoints:
            raise click.ClickException(f"No .ckpt files found in required checkpoint directory: {variant_dir}")
        checkpoint_groups.extend((variant, checkpoint) for checkpoint in checkpoints)

    async def get_dataset() -> Dataset:
        async with get_async_db_session_ctx() as session:
            return await DatasetService(session).get_dataset_by_id(dataset_id)

    async def persist_import(model_dir: Path, model_name: str, snapflow_enabled: bool) -> Model:
        job = TrainJob(
            project_id=project_id,
            payload=LocalTrainJobPayload(
                project_id=project_id,
                dataset_id=dataset_id,
                policy=policy,
                model_name=model_name,
                max_steps=100,
                batch_size=1,
                auto_scale_batch_size=False,
                val_split=0.1,
                device=None,
            ),
            status=JobStatus.COMPLETED,
            message="WebSummit checkpoint import completed",
        )
        model = Model(
            id=UUID(model_dir.name),
            project_id=project_id,
            dataset_id=dataset_id,
            path=str(model_dir),
            name=model_name,
            snapshot_id=None,
            policy=policy,
            properties={SNAPFLOW_PROPERTY: snapflow_enabled},
            train_job_id=job.id,
            parent_model_id=None,
            version=1,
            created_at=None,
        )
        async with get_async_db_session_ctx() as session:
            saved_job = await JobService(session).create_job(job)
            return await ModelService(session).create_model(model.model_copy(update={"train_job_id": saved_job.id}))

    dataset = asyncio.run(get_dataset())
    if dataset.project_id != project_id:
        raise click.ClickException("Dataset does not belong to the specified project")

    settings = get_settings()
    failures: list[str] = []
    for variant, checkpoint in checkpoint_groups:
        model_name = f"{variant} {checkpoint.stem}"
        model_dir = settings.models_dir / str(uuid4())
        destination = model_dir / "model.ckpt"
        snapflow_enabled = checkpoint.name.startswith("snapflow-")
        click.echo(f"Importing {checkpoint} as '{model_name}'")

        try:
            model_dir.mkdir(parents=True)
            shutil.move(str(checkpoint), str(destination))
            model = asyncio.run(persist_import(model_dir, model_name, snapflow_enabled))
        except Exception as exc:
            if destination.is_file() and not checkpoint.exists():
                shutil.move(str(destination), str(checkpoint))
            shutil.rmtree(model_dir, ignore_errors=True)
            failures.append(f"{checkpoint.name}: import failed ({exc})")
            continue

        def report(_progress: int, message: str | None, _extra_info: dict) -> None:
            if message is not None:
                click.echo(f"{model_name}: {message}")

        try:
            loaded_policy = _load_policy_from_checkpoint(TrainingJobSpec(policy=policy), destination)
            _export(loaded_policy, model_dir, report, requested=["torch", "openvino"])
            click.echo(f"Imported model {model.id}")
        except Exception as exc:
            # The checkpoint and model record remain available for a later
            # `models export` retry when one backend's dependencies fail.
            failures.append(f"{checkpoint.name}: export failed for model {model.id} ({exc})")

    if failures:
        raise click.ClickException("WebSummit import completed with failures:\n" + "\n".join(failures))
