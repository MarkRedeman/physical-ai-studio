from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from core.logging.utils import job_logging_ctx
from db import get_async_db_session_ctx
from schemas.base_job import JobStatus
from schemas.export_job import ModelExportJob, ModelExportJobPayload
from schemas.job import Job
from services.event_processor import EventType
from services.job_service import JobService
from services.model_export_service import ModelExportService
from workers.base import BaseProcessWorker

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass
    from uuid import UUID

    from physicalai.export.backends import ExportBackend


class ModelExportWorker(BaseProcessWorker):
    """Process model re-export jobs (torch + OpenVINO + optional NNCF compression).

    Re-exports run in a separate process because they load a policy checkpoint
    and run torch/OpenVINO/NNCF conversions, which would otherwise block the API.
    """

    ROLE = "ModelExportWorker"

    def __init__(self, stop_event: EventClass, event_queue: mp.Queue):
        super().__init__(stop_event=stop_event)
        self.queue = event_queue

    async def run_loop(self) -> None:
        logger.info("Model Export Worker is running")
        while not self.should_stop():
            async with get_async_db_session_ctx() as session:
                pending = await JobService(session).get_pending_model_export_jobs()
            for job in pending:
                await self._process_job(job)
            self.stop_aware_sleep(0.5)

    async def _process_job(self, job: Job) -> None:
        with job_logging_ctx(job_id=str(job.id)):
            job = ModelExportJob.model_validate(job.model_dump())
            payload = ModelExportJobPayload.model_validate(job.payload)
            logger.info(
                "Processing model export job: job_id='{}', model_id='{}', backends={}, compress={}",
                job.id,
                payload.model_id,
                payload.backends,
                payload.compress,
            )

            updated = await self._update_job(
                job.id,
                status=JobStatus.RUNNING,
                message="Exporting model",
                progress=5,
            )
            self.queue.put((EventType.JOB_UPDATE, updated))

            try:
                backends: list[str | ExportBackend] = list(payload.backends)
                exported = await ModelExportService.export_model(
                    payload.model_id,
                    name=payload.name,
                    backends=backends,
                    compress=payload.compress,
                )
                completed = await self._update_job(
                    job.id,
                    status=JobStatus.COMPLETED,
                    message="Model exported",
                    progress=100,
                    extra_info={"exported_model_id": str(exported.id)},
                )
                logger.info("Model export job completed: job_id='{}', exported_model_id='{}'", job.id, exported.id)
                self.queue.put((EventType.JOB_UPDATE, completed))
            except Exception as exc:
                logger.exception("Model export failed for job {}: {}", job.id, exc)
                failed = await self._update_job(
                    job.id,
                    status=JobStatus.FAILED,
                    message=f"Model export failed: {exc}",
                )
                self.queue.put((EventType.JOB_UPDATE, failed))

    @staticmethod
    async def _update_job(
        job_id: UUID,
        *,
        status: JobStatus,
        message: str,
        progress: int | None = None,
        extra_info: dict | None = None,
    ) -> ModelExportJob:
        async with get_async_db_session_ctx() as session:
            job = await JobService(session).update_job_status(
                job_id,
                status=status,
                message=message,
                progress=progress,
                extra_info=extra_info,
            )
        return ModelExportJob.model_validate(job)
