from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from core.logging.utils import job_logging_ctx
from schemas.base_job import JobStatus
from schemas.dataset_import_job import DatasetImportJobPayload, ImportStep, ImportValidationSeverity
from schemas.job import DatasetImportJob
from services.archive_safety import SafeZipArchive, cleanup_staged_archive
from services.dataset_import.adapters import (
    DatasetImportAdapter,
    get_registered_dataset_import_adapters,
    select_dataset_import_adapter,
)
from services.dataset_import.service import DatasetImportService
from services.dataset_import.staging import resolve_payload_archive_path
from services.event_processor import EventType
from services.job_service import JobService
from settings import get_settings
from workers.base import BaseProcessWorker

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass
    from uuid import UUID


class DatasetImportWorker(BaseProcessWorker):
    ROLE = "DatasetImportWorker"

    def __init__(self, stop_event: EventClass, event_queue: mp.Queue):
        super().__init__(stop_event=stop_event)
        self.queue = event_queue
        self.adapters: list[DatasetImportAdapter] = get_registered_dataset_import_adapters()

    async def run_loop(self) -> None:
        logger.info("Dataset Import Worker is running")
        while not self.should_stop():
            job = await DatasetImportService.claim_pending_dataset_import_job()
            if isinstance(job, DatasetImportJob):
                await self._process_job(job)
            await asyncio.sleep(0.5)

    async def _process_job(self, job: DatasetImportJob) -> None:
        with job_logging_ctx(job_id=str(job.id)):
            if not isinstance(job.payload, DatasetImportJobPayload):
                raise ValueError(f"Invalid payload type for dataset import job: {type(job.payload)}")
            payload = job.payload
            archive_ref = payload.archive_staging_id
            logger.info(
                "Processing dataset import job: job_id='{}', step='{}', source_hint='{}', archive_ref='{}'",
                job.id,
                payload.step,
                payload.source_hint,
                archive_ref,
            )

            pre_commit_handled = False
            try:
                if payload.step == ImportStep.UPLOADED:
                    await self._run_detection(job.id, job.project_id, payload)
                elif payload.step == ImportStep.READY_TO_COMMIT:
                    pre_commit_handled = await self._run_commit(job.id, job.project_id, payload)
            except Exception as e:
                logger.exception(f"Dataset import failed: {e}")
                if not pre_commit_handled:
                    failed_job = await JobService.update_job_status(
                        job_id=job.id,
                        status=JobStatus.FAILED,
                        message=f"Dataset import failed: {e}",
                    )
                    cleanup_staged_archive(resolve_payload_archive_path(payload))
                    self.queue.put((EventType.JOB_UPDATE, failed_job))

    async def _run_detection(self, job_id: UUID, _project_id: UUID, payload: DatasetImportJobPayload) -> None:
        archive_path = resolve_payload_archive_path(payload)

        logger.info(
            "Starting source detection for job_id='{}' with staging_id='{}', archive='{}'",
            job_id,
            payload.archive_staging_id,
            archive_path,
        )
        payload.step = ImportStep.DETECTING_SOURCE
        job = await JobService.update_job_payload(
            job_id=job_id,
            payload=payload,
            status=JobStatus.RUNNING,
            message="Detecting dataset import source",
            progress=10,
        )
        self.queue.put((EventType.JOB_UPDATE, job))

        archive = SafeZipArchive(
            archive_path,
            max_uncompressed_bytes=get_settings().data_import_max_uncompressed_bytes,
        )
        selected_adapter = select_dataset_import_adapter(
            adapters=self.adapters,
            source_hint=payload.source_hint,
            archive=archive,
        )
        if selected_adapter is None:
            logger.error("No dataset import adapter detected for archive='{}'", archive.path)
            raise ValueError("Unable to detect source adapter for dataset archive")

        logger.info(
            "Auto-detected dataset format: adapter='{}', source='{}', archive='{}'",
            selected_adapter.__class__.__name__,
            selected_adapter.source,
            archive.path,
        ) if payload.source_hint == "auto" else logger.info(
            "Selecting adapter from source hint: hint='{}', adapter='{}'",
            payload.source_hint,
            selected_adapter.__class__.__name__,
        )

        payload.dataset_manifest_draft, report = selected_adapter.build_draft(archive=archive, payload=payload)
        payload.dataset_manifest_draft.source_type = selected_adapter.source

        logger.info(
            "Dataset source decided for job_id='{}': source='{}'",
            job_id,
            payload.dataset_manifest_draft.source_type,
        )
        payload.step = ImportStep.GENERATING_DRAFT_MANIFEST
        job = await JobService.update_job_payload(
            job_id=job_id,
            payload=payload,
            status=JobStatus.RUNNING,
            message="Generating dataset draft manifest",
            progress=25,
        )
        self.queue.put((EventType.JOB_UPDATE, job))

        error_count = sum(1 for msg in report.messages if msg.severity == ImportValidationSeverity.ERROR)
        warning_count = sum(1 for msg in report.messages if msg.severity == ImportValidationSeverity.WARNING)
        logger.info(
            "Pre-finalize validation report for job_id='{}': is_valid={}, errors={}, warnings={}",
            job_id,
            report.is_valid,
            error_count,
            warning_count,
        )
        # Only persist validation_report when there is actionable content; leave None when clean.
        has_issues = bool(report.messages)
        payload.validation_report = report if has_issues else None
        payload.step = ImportStep.WAITING_FOR_USER_INPUT

        job = await JobService.update_job_payload(
            job_id=job_id,
            payload=payload,
            status=JobStatus.PENDING,
            message="Dataset import ready for user finalization",
            progress=40,
        )
        self.queue.put((EventType.JOB_UPDATE, job))

    async def _run_commit(self, job_id: UUID, project_id: UUID, payload: DatasetImportJobPayload) -> bool:
        """Run the commit phase.  Returns True if this method handled the final job state
        (pre-commit validation failure), so the outer error handler does not double-mark.
        """
        archive_path = resolve_payload_archive_path(payload)

        if payload.dataset_manifest_draft is None:
            raise ValueError("No dataset manifest draft found; cannot determine adapter for commit")

        expected_source_type = payload.dataset_manifest_draft.source_type
        adapter = next((a for a in self.adapters if a.source == expected_source_type), None)
        if adapter is None:
            raise ValueError(f"No adapter available for source_type='{expected_source_type}' during commit")

        logger.info(
            "Starting commit for job_id='{}' using adapter='{}', source='{}', staging_id='{}'",
            job_id,
            adapter.__class__.__name__,
            expected_source_type,
            payload.archive_staging_id,
        )

        pre_commit_report = adapter.validate_pre_commit(payload=payload)
        if not pre_commit_report.is_valid:
            # Persist failure state with the blocking report so it is observable in the payload
            error_messages = "; ".join(
                msg.message for msg in pre_commit_report.messages if msg.severity == ImportValidationSeverity.ERROR
            )
            payload.validation_report = pre_commit_report
            failed_job = await JobService.update_job_payload(
                job_id=job_id,
                payload=payload,
                status=JobStatus.FAILED,
                message=f"Pre-commit validation failed: {error_messages}",
            )
            cleanup_staged_archive(archive_path)
            self.queue.put((EventType.JOB_UPDATE, failed_job))
            logger.warning(
                "Pre-commit validation failed for job_id='{}': {}",
                job_id,
                error_messages,
            )
            return True  # handled; outer handler must not double-mark

        try:
            payload.step = ImportStep.IMPORTING_RESOURCE
            job = await JobService.update_job_payload(
                job_id=job_id,
                payload=payload,
                status=JobStatus.RUNNING,
                message="Starting dataset import",
                progress=60,
            )
            self.queue.put((EventType.JOB_UPDATE, job))

            archive = SafeZipArchive(
                archive_path,
                max_uncompressed_bytes=get_settings().data_import_max_uncompressed_bytes,
            )
            dataset = await adapter.commit(payload, project_id=project_id, archive=archive)
            logger.info(
                "Adapter commit completed for job_id='{}': dataset_id='{}', dataset_path='{}'",
                job_id,
                dataset.id,
                dataset.path,
            )
            payload.result_dataset_id = dataset.id
            payload.step = ImportStep.REGISTERING_RESOURCE
            job = await JobService.update_job_payload(
                job_id=job_id,
                payload=payload,
                status=JobStatus.RUNNING,
                message="Registering dataset import result",
                progress=90,
            )
            self.queue.put((EventType.JOB_UPDATE, job))

            payload.step = ImportStep.COMPLETED
            completed = await JobService.update_job_payload(
                job_id=job_id,
                payload=payload,
                status=JobStatus.COMPLETED,
                message="Dataset import completed",
                progress=100,
            )
            logger.info(
                "Dataset import job completed: job_id='{}', result_dataset_id='{}'", job_id, payload.result_dataset_id
            )
            self.queue.put((EventType.JOB_UPDATE, completed))
            return False
        finally:
            cleanup_staged_archive(archive_path)
