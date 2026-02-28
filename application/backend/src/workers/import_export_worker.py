from __future__ import annotations

import asyncio
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass

from loguru import logger

from core.logging.utils import job_logging_ctx
from schemas.job import ExportJob, ExportJobPayload, ImportJob, ImportJobPayload, JobStatus, JobType
from services import JobService, ModelService
from services.event_processor import EventType
from workers.base import BaseProcessWorker


class ImportExportWorker(BaseProcessWorker):
    ROLE = "ImportExportWorker"

    def __init__(self, stop_event: EventClass, event_queue: mp.Queue):
        super().__init__(stop_event=stop_event)
        self.queue = event_queue

    async def run_loop(self) -> None:
        job_service = JobService()
        logger.info("ImportExport Worker is running")
        while not self.should_stop():
            job = await job_service.get_pending_import_export_job()
            if job is not None:
                with job_logging_ctx(job_id=str(job.id)):
                    if job.type == JobType.IMPORT:
                        await self._handle_import(job)
                    elif job.type == JobType.EXPORT:
                        await self._handle_export(job)
            await asyncio.sleep(0.5)

    def setup(self) -> None:
        super().setup()
        with logger.contextualize(worker=self.__class__.__name__):
            if self.loop is None:
                raise RuntimeError("The event loop must be set.")
            self.loop.run_until_complete(self._abort_orphan_jobs())

    def teardown(self) -> None:
        super().teardown()
        with logger.contextualize(worker=self.__class__.__name__):
            if self.loop is None:
                raise RuntimeError("The event loop must be set.")
            self.loop.run_until_complete(self._abort_orphan_jobs())

    @staticmethod
    async def _abort_orphan_jobs() -> None:
        """Mark any RUNNING or PENDING import/export jobs as FAILED on startup/shutdown."""
        from db import get_async_db_session_ctx
        from repositories import JobRepository

        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            running_jobs = await repo.get_all(
                extra_filters={"status": JobStatus.RUNNING},
            )
            pending_jobs = await repo.get_all(
                extra_filters={"status": JobStatus.PENDING},
            )

            for job in running_jobs + pending_jobs:
                if job.type in (JobType.IMPORT, JobType.EXPORT):
                    await repo.update(
                        job,
                        {
                            "status": JobStatus.FAILED,
                            "message": "Job aborted: worker restarted",
                        },
                    )

    async def _handle_import(self, job: ImportJob) -> None:
        """Handle a model import job."""
        updated_job = await JobService.update_job_status(
            job_id=job.id, status=JobStatus.RUNNING, message="Importing model..."
        )
        self.queue.put((EventType.JOB_UPDATE, updated_job))

        try:
            payload = job.payload

            # Read the uploaded file from disk
            file_path = Path(payload.upload_file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Upload file not found: {file_path}")

            file_content = file_path.read_bytes()

            # Create ModelImportService (same dependencies as the DI factory)
            from services.dataset_service import DatasetService
            from services.model_import_service import ModelImportService
            from services.model_service import ModelService
            from services.snapshot_service import SnapshotService

            model_import_service = ModelImportService(
                model_service=ModelService(),
                dataset_service=DatasetService(),
                snapshot_service=SnapshotService(),
            )

            model = await model_import_service.import_model(
                file_content=file_content,
                filename=payload.original_filename,
                name=payload.model_name,
                project_id=str(payload.project_id),
            )

            updated_job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.COMPLETED, message="Import finished"
            )
            self.queue.put((EventType.MODEL_UPDATE, model))

            # Clean up temp file
            try:
                file_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"Failed to clean up temp file: {file_path}")

        except Exception as e:
            logger.exception(f"Import failed: {e}")
            updated_job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.FAILED, message=f"Import failed: {e}"
            )

        self.queue.put((EventType.JOB_UPDATE, updated_job))

    async def _handle_export(self, job: ExportJob) -> None:
        """Handle a model export job."""
        updated_job = await JobService.update_job_status(
            job_id=job.id, status=JobStatus.RUNNING, message="Exporting model..."
        )
        self.queue.put((EventType.JOB_UPDATE, updated_job))

        try:
            payload = job.payload

            model = await ModelService.get_model_by_id(payload.model_id)
            model_path = Path(model.path).expanduser()
            exports_path = model_path / "exports"

            if not exports_path.is_dir():
                raise FileNotFoundError(f"Model exports directory not found: {exports_path}")

            from settings import get_settings

            settings = get_settings()

            manifest = {
                "name": model.name,
                "policy": model.policy,
                "properties": model.properties,
                "original_model_id": str(model.id),
                "original_project_id": str(model.project_id),
                "original_dataset_id": str(model.dataset_id) if model.dataset_id else None,
                "original_snapshot_id": str(model.snapshot_id) if model.snapshot_id else None,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "physicalai_version": settings.version,
            }

            # Build zip to disk
            archive_dir = model_path / "exports" / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            safe_name = model.name.replace('"', "").replace("/", "_").replace("\\", "_")
            zip_path = archive_dir / f"{safe_name}.zip"

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Add manifest
                zf.writestr(
                    str(Path(model.name) / "manifest.json"),
                    json.dumps(manifest, indent=2),
                )
                # Add export artifacts
                for file_path in sorted(exports_path.rglob("*")):
                    if file_path.is_file() and "archive" not in file_path.parts:
                        arcname = str(Path(model.name) / file_path.relative_to(model_path))
                        zf.write(file_path, arcname)

            updated_job = await JobService.update_job_status(
                job_id=job.id,
                status=JobStatus.COMPLETED,
                message="Export finished",
                extra_info={"zip_path": str(zip_path)},
            )
        except Exception as e:
            logger.exception(f"Export failed: {e}")
            updated_job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.FAILED, message=f"Export failed: {e}"
            )

        self.queue.put((EventType.JOB_UPDATE, updated_job))
