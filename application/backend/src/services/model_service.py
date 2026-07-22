import shutil
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from loguru import logger

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories import ModelRepository, SnapshotRepository
from schemas.job import TrainJob
from schemas.model import BackendExportDetail, Model, TrainingSummary


class SafeLoaderLegacyPythonTags(yaml.SafeLoader):
    """Safe loader with support for legacy ``!!python/*`` YAML tags."""


def _construct_legacy_python_tag(
    loader: yaml.SafeLoader,
    tag_suffix: str,
    node: yaml.Node,
) -> Any:
    # pyrefly/mypy: tag_suffix is intentionally unused; constructor signature is fixed by PyYAML.
    del tag_suffix
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    return None


SafeLoaderLegacyPythonTags.add_multi_constructor("tag:yaml.org,2002:python/", _construct_legacy_python_tag)


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
            detail = BackendExportDetail.from_backend_dir(backend_dir)
            if detail is not None:
                details.append(detail)

        return details

    @staticmethod
    def get_hparams(model: Model) -> dict | None:
        """Read training hyperparameters from the model directory.

        Looks for ``version_0/hparams.yaml`` (written by Lightning's CSVLogger).
        """
        hparams_path = Path(model.path) / "version_0" / "hparams.yaml"
        if not hparams_path.is_file():
            return None
        try:
            with hparams_path.open(encoding="utf-8") as f:
                loader = SafeLoaderLegacyPythonTags(f)
                try:
                    data = loader.get_single_data()
                finally:
                    loader.dispose()
        except yaml.YAMLError:
            logger.warning("Could not parse model hparams YAML at {}", hparams_path)
            return None

        return data if isinstance(data, dict) else None

    @staticmethod
    def get_training_summary(training_job: TrainJob | None) -> TrainingSummary | None:
        """Extract a summary of training configuration from a training job.

        This merges fields from the job's payload (batch size, precision, etc.)
        with computed values like training duration.
        """
        if training_job is None:
            return None

        payload = training_job.payload
        device_type = str(payload.device.type) if payload.device is not None else None

        return TrainingSummary(
            max_steps=payload.max_steps,
            batch_size=payload.batch_size,
            precision=str(payload.precision),
            compile_model=payload.compile_model,
            val_split=payload.val_split,
            auto_scale_batch_size=payload.auto_scale_batch_size,
            num_workers=payload.num_workers,
            device_type=device_type,
        )
