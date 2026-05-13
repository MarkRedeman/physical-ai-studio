import asyncio
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from exceptions import InvalidArchiveError
from schemas import Model, TrainJob
from schemas.base_job import JobStatus
from schemas.job import TrainJobPayload
from services.archive_safety import SafeZipArchive, flatten_single_root_directory
from services.dataset_service import DatasetService
from services.job_service import JobService
from services.model_service import ModelService
from services.snapshot_service import SnapshotService
from settings import get_settings


class ModelImportService:
    async def import_model_archive(
        self,
        *,
        archive_path: Path,
        project_id: UUID,
        dataset_id: UUID,
        policy: str,
        model_name: str,
        base_model_id: UUID | None = None,
        version: int = 1,
    ) -> Model:
        if not archive_path.exists() or not archive_path.is_file():
            raise InvalidArchiveError(f"Model archive does not exist: {archive_path}")

        settings = get_settings()
        dataset = await DatasetService.get_dataset_by_id(dataset_id)
        model_id = uuid4()
        model_dir = settings.models_dir / str(model_id)

        safe_archive = SafeZipArchive(
            archive_path,
            max_uncompressed_bytes=settings.data_import_max_uncompressed_bytes,
        )
        await asyncio.to_thread(safe_archive.validate)

        model_dir.mkdir(parents=True, exist_ok=False)
        try:
            extracted_count = await asyncio.to_thread(
                safe_archive.extract_to,
                model_dir,
                min_free_bytes=settings.data_import_min_free_bytes,
            )
            await asyncio.to_thread(flatten_single_root_directory, model_dir)

            if extracted_count == 0:
                raise InvalidArchiveError("Model archive does not contain any files")

            snapshot_dir = model_dir / SnapshotService.generate_snapshot_folder_name()
            snapshot = await SnapshotService.create_snapshot_for_dataset(dataset, destination=snapshot_dir)

            job = TrainJob(
                project_id=project_id,
                payload=TrainJobPayload(
                    project_id=project_id,
                    dataset_id=dataset_id,
                    policy=policy,
                    model_name=model_name,
                    max_steps=100,
                    batch_size=1,
                    auto_scale_batch_size=False,
                    base_model_id=base_model_id,
                    val_split=0.1,
                    device=None,
                ),
                status=JobStatus.COMPLETED,
                message="Model import completed",
            )
            job = await JobService.create_job(job)

            model = Model(
                id=model_id,
                project_id=project_id,
                dataset_id=dataset_id,
                path=str(model_dir),
                name=model_name,
                snapshot_id=snapshot.id,
                policy=policy,
                properties={},
                train_job_id=job.id,
                parent_model_id=base_model_id,
                version=version,
                created_at=None,
            )
            return await ModelService.create_model(model)
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise
