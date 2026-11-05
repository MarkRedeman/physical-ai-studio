import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.schema import JobDB
from exceptions import InvalidResourceError, ResourceInUseError, ResourceNotFoundError, ResourceType
from repositories import ModelRepository, SnapshotRepository
from schemas.hardware import InferenceBackend
from schemas.job import TrainJob
from schemas.model import BackendExportDetail, BackendIOSpec, ExportVariantDetail, Model, TrainingSummary
from schemas.model_variant import export_directory, package_files, read_package, variant_info


class ModelService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = ModelRepository(session)

    async def get_model_list(self) -> list[Model]:
        return await self._publication_state(await self.repo.get_all())

    async def get_model_by_id(self, model_id: UUID) -> Model:
        model = await self.repo.get_by_id(model_id)
        if model is None:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        return (await self._publication_state([model]))[0]

    async def _publication_state(self, models: list[Model]) -> list[Model]:
        # A worker can die after committing completion but before removing its
        # publication marker. Reconcile lazily, without a new database schema.
        for model in models:
            for marker in (Path(model.path) / "exports").glob("*/variants/*/.unpublished"):
                try:
                    job_id = UUID(marker.read_text(encoding="utf-8"))
                    backend, variant = marker.parent.parent.parent.name, marker.parent.name
                    directory = export_directory(Path(model.path), backend, variant)
                    metadata = variant_info(read_package(directory, backend), variant)
                    job = await self.session.scalar(select(JobDB).where(JobDB.id == str(job_id)))
                    payload = cast("dict[str, object]", job.payload) if job is not None else {}
                    if (
                        job is not None
                        and job.status == "completed"
                        and metadata.job_id == job_id
                        and payload.get("model_id") == str(model.id)
                        and payload.get("backend", "openvino") == backend
                        and payload.get("target_variant", "int8") == variant
                    ):
                        marker.unlink(missing_ok=True)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return models

    async def create_model(self, model: Model) -> Model:
        return await self.repo.save(model)

    async def update_model(self, model: Model, update: dict) -> Model:
        return await self.repo.update(model, update)

    async def delete_model(self, model: Model) -> None:
        await self._require_no_active_variant_jobs(model)
        snapshot_repo = SnapshotRepository(self.session)
        await self.repo.delete_by_id(model.id)

        if model.snapshot_id is not None:
            await snapshot_repo.delete_by_id(model.snapshot_id)

        model_path = Path(model.path).expanduser()
        shutil.rmtree(model_path)

    async def delete_variant(self, model: Model, backend: str, variant: str) -> None:
        """Remove an unused named package, never the original backend export."""
        if variant == "original":
            raise InvalidResourceError("Model", "The original export cannot be deleted independently")
        directory = self.resolve_export(model, backend, variant)
        await self._require_no_active_variant_jobs(model)
        shutil.rmtree(directory)

    async def _require_no_active_variant_jobs(self, model: Model) -> None:
        active = await self.session.scalar(
            select(JobDB.id)
            .where(
                JobDB.status.in_(("pending", "running")),
                func.json_extract(JobDB.payload, "$.model_id") == str(model.id),
            )
            .limit(1)
        )
        if active is not None:
            raise ResourceInUseError(ResourceType.MODEL, str(model.id))

    async def get_project_models(self, project_id: UUID) -> list[Model]:
        return await self._publication_state(await self.repo.get_project_models(project_id))

    @staticmethod
    def get_backend_details(model: Model) -> list[BackendExportDetail]:
        """Compute per-backend export details from the filesystem."""
        exports_dir = Path(model.path) / "exports"
        if not exports_dir.is_dir():
            return []

        from services.model_optimization_service import ModelOptimizationService

        grouped: dict[str, list[ExportVariantDetail]] = {}
        packages = model.export_packages()
        for (backend, variant), directory in sorted(
            packages.items(), key=lambda item: (item[0][0], item[0][1] != "original", item[0][1])
        ):
            manifest = read_package(directory, backend)
            info = variant_info(manifest, variant)
            # Legacy roots can already contain distilled weights.
            if variant == "original" and model.snapflow_enabled and "variant" not in manifest.get("studio", {}):
                info.checkpoint_kind = "snapflow"
            files = package_files(directory)
            grouped.setdefault(backend, []).append(
                ExportVariantDetail(
                    **info.model_dump(),
                    label="Original (SnapFlow)"
                    if variant == "original" and info.checkpoint_kind == "snapflow"
                    else "Original"
                    if variant == "original"
                    else " + ".join(
                        part.upper() if part == "rtc" else "INT8" if part == "int8" else "SnapFlow"
                        for part in variant.split("-")
                    ),
                    size_bytes=sum(path.stat().st_size for path in files),
                    file_count=len(files),
                    exported_at=info.created_at
                    or datetime.fromtimestamp((directory / "manifest.json").stat().st_mtime, tz=UTC),
                    io_spec=BackendIOSpec.from_manifest(manifest),
                    can_run_in_studio=backend in {supported.value for supported in InferenceBackend},
                    available_operations=ModelOptimizationService.operations(model, backend, info, packages),
                )
            )
        return [
            BackendExportDetail(
                type=backend,
                size_bytes=next((v.size_bytes for v in variants if v.id == "original"), 0),
                file_count=next((v.file_count for v in variants if v.id == "original"), 0),
                exported_at=next((v.exported_at for v in variants if v.id == "original"), None),
                io_spec=next((v.io_spec for v in variants if v.id == "original"), None),
                variants=variants,
            )
            for backend, variants in sorted(grouped.items())
        ]

    @staticmethod
    def resolve_export(model: Model, backend: str, variant: str = "original") -> Path:
        directory = model.export_packages().get((backend, variant))
        if directory is None:
            raise ResourceNotFoundError(ResourceType.MODEL, f"{model.id}/{backend}/{variant}")
        return directory

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
    def get_training_summary(training_job: TrainJob | None) -> TrainingSummary | None:
        """Extract a summary of training configuration from a training job.

        This merges fields from the job's payload (batch size, precision, etc.)
        with computed values like training duration.
        """
        if training_job is None:
            return None

        payload = training_job.payload
        device_type = str(payload.device.type) if payload.device is not None else None
        lora_enabled = payload.lora_enabled
        snapflow_distill_epochs = payload.snapflow_distill_epochs if payload.snapflow_enabled else None

        return TrainingSummary(
            max_epochs=payload.max_epochs,
            max_steps=payload.max_steps,
            batch_size=payload.batch_size,
            precision=str(payload.precision),
            compile_model=payload.compile_model,
            augment_images=payload.augment_images,
            val_split=payload.val_split,
            auto_scale_batch_size=payload.auto_scale_batch_size,
            num_workers=payload.num_workers,
            device_type=device_type,
            lora_enabled=lora_enabled,
            lora_rank=payload.lora_rank if lora_enabled else None,
            lora_alpha=payload.lora_alpha if lora_enabled else None,
            lora_dropout=payload.lora_dropout if lora_enabled else None,
            lora_use_dora=payload.lora_use_dora if lora_enabled else None,
            snapflow_enabled=payload.snapflow_enabled,
            snapflow_distill_epochs=snapflow_distill_epochs,
        )
