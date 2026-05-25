import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import yaml

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories import ModelRepository, SnapshotRepository
from schemas.model import BackendExportDetail, Model, TrainingSummary

if TYPE_CHECKING:
    from schemas.job import TrainJob


class ModelService:
    @staticmethod
    async def get_model_list() -> list[Model]:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.get_all()

    @staticmethod
    async def get_model_by_id(model_id: UUID) -> Model:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            model = await repo.get_by_id(model_id)
            if model is None:
                raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

            return model

    @staticmethod
    async def create_model(model: Model) -> Model:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.save(model)

    @staticmethod
    async def update_model(model: Model, update: dict) -> Model:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.update(model, update)

    @staticmethod
    async def delete_model(model: Model) -> None:
        async with get_async_db_session_ctx() as session:
            model_repo = ModelRepository(session)
            snapshot_repo = SnapshotRepository(session)

            await model_repo.delete_by_id(model.id)

            # Remove the associated snapshot row to avoid stale FK references
            if model.snapshot_id is not None:
                await snapshot_repo.delete_by_id(model.snapshot_id)

        model_path = Path(model.path).expanduser()
        shutil.rmtree(model_path)

    @staticmethod
    async def get_project_models(project_id: UUID) -> list[Model]:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.get_project_models(project_id)

    @staticmethod
    def get_backend_details(model: Model) -> list[BackendExportDetail]:
        """Compute per-backend export details from the filesystem."""
        exports_dir = Path(model.path) / "exports"
        if not exports_dir.is_dir():
            return []

        details: list[BackendExportDetail] = []
        for backend_dir in sorted(exports_dir.iterdir()):
            if not backend_dir.is_dir():
                continue
            files = [f for f in backend_dir.rglob("*") if f.is_file()]

            # Backend exports folder may be empty if export failed
            if len(files) == 0:
                continue

            total_size = sum(f.stat().st_size for f in files)
            exported_at = datetime.fromtimestamp(backend_dir.stat().st_mtime)
            details.append(
                BackendExportDetail(
                    type=backend_dir.name,
                    size_bytes=total_size,
                    file_count=len(files),
                    exported_at=exported_at,
                )
            )
        return details

    @staticmethod
    def get_hparams(model: Model) -> dict | None:
        """Read training hyperparameters from the model directory.

        Looks for ``version_0/hparams.yaml`` (written by Lightning's CSVLogger).
        """
        hparams_path = Path(model.path) / "version_0" / "hparams.yaml"
        if not hparams_path.is_file():
            return None
        with hparams_path.open() as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_training_summary(training_job: "TrainJob | None") -> TrainingSummary | None:
        """Extract a summary of training configuration from a training job.

        This merges fields from the job's payload (batch size, precision, etc.)
        with computed values like training duration.
        """
        if training_job is None:
            return None

        payload = training_job.payload

        duration = None
        if training_job.start_time is not None and training_job.end_time is not None:
            duration = (training_job.end_time - training_job.start_time).total_seconds()

        device_type = None
        if payload.device is not None:
            device_type = str(payload.device.type)

        return TrainingSummary(
            max_steps=payload.max_steps,
            batch_size=payload.batch_size,
            precision=str(payload.precision),
            compile_model=payload.compile_model,
            val_split=payload.val_split,
            auto_scale_batch_size=payload.auto_scale_batch_size,
            num_workers=payload.num_workers,
            device_type=device_type,
            training_duration_seconds=duration,
        )
