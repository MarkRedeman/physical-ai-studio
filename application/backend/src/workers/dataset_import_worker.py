from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from core.logging.utils import job_logging_ctx
from schemas.base_job import JobStatus
from schemas.import_job import DatasetImportJobPayload, ImportStep
from services.dataset_import_service import DatasetImportService
from services.event_processor import EventType
from services.import_adapters import (
    DatasetImportAdapter,
    LeRobotV2Adapter,
    LeRobotV3Adapter,
    StudioAdapter,
    TrossenSDKAdapter,
)
from services.job_service import JobService
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
        self.adapters: list[DatasetImportAdapter] = [
            StudioAdapter(),
            LeRobotV2Adapter(),
            LeRobotV3Adapter(),
            TrossenSDKAdapter(),
        ]

    def _pick_adapter(self, payload: DatasetImportJobPayload, archive_path: Path) -> DatasetImportAdapter | None:
        if payload.source_hint != "auto":
            hinted = next((adapter for adapter in self.adapters if adapter.source == payload.source_hint), None)
            if hinted is not None:
                logger.info(
                    "Selecting adapter from source hint: hint='{}', adapter='{}'",
                    payload.source_hint,
                    hinted.__class__.__name__,
                )
                if not hinted.detect(archive_path):
                    raise ValueError(
                        f"The uploaded archive is not compatible with the selected format '{payload.source_hint}'"
                    )
                return hinted
            logger.warning("Source hint '{}' provided, but no matching adapter was found", payload.source_hint)

        for adapter in self.adapters:
            if adapter.detect(archive_path):
                logger.info(
                    "Auto-detected dataset format: adapter='{}', source='{}', archive='{}'",
                    adapter.__class__.__name__,
                    adapter.source,
                    archive_path,
                )
                return adapter

        logger.error("No dataset import adapter detected for archive='{}'", archive_path)
        return None

    async def run_loop(self) -> None:
        logger.info("Dataset Import Worker is running")
        while not self.should_stop():
            job = await DatasetImportService.get_pending_dataset_import_job()
            if job is not None:
                await self._process_job(job.id)
            await asyncio.sleep(0.5)

    async def _process_job(self, job_id: UUID) -> None:
        with job_logging_ctx(job_id=str(job_id)):
            job = await JobService.get_job_by_id(job_id)
            if not isinstance(job.payload, DatasetImportJobPayload):
                raise ValueError(f"Invalid payload type for dataset import job: {type(job.payload)}")
            payload = job.payload
            logger.info(
                "Processing dataset import job: job_id='{}', step='{}', source_hint='{}', archive='{}'",
                job_id,
                payload.step,
                payload.source_hint,
                payload.uploaded_archive_path,
            )

            try:
                if payload.step == ImportStep.UPLOADED:
                    await self._run_detection(job_id, job.project_id, payload)
                elif payload.step == ImportStep.READY_TO_COMMIT:
                    await self._run_commit(job_id, job.project_id, payload)
            except Exception as e:
                logger.exception(f"Dataset import failed: {e}")
                failed_job = await JobService.update_job_status(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    message=f"Dataset import failed: {e}",
                )
                self.queue.put((EventType.JOB_UPDATE, failed_job))

    async def _run_detection(self, job_id: UUID, project_id: UUID, payload: DatasetImportJobPayload) -> None:
        logger.info(
            "Starting source detection for job_id='{}' with archive='{}'",
            job_id,
            payload.uploaded_archive_path,
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

        archive_path = Path(payload.uploaded_archive_path)
        selected_adapter = self._pick_adapter(payload, archive_path)

        if selected_adapter is None:
            raise ValueError("Unable to detect source adapter for dataset archive")

        if payload.dataset_manifest_draft is None:
            payload.dataset_manifest_draft = selected_adapter.parse_to_draft_manifest(
                archive_path=archive_path, payload=payload
            )

        payload.dataset_manifest_draft.source.source_type = selected_adapter.source
        payload.dataset_manifest_draft.source.adapter = selected_adapter.__class__.__name__
        logger.info(
            "Dataset source decided for job_id='{}': source='{}', adapter='{}'",
            job_id,
            payload.dataset_manifest_draft.source.source_type,
            payload.dataset_manifest_draft.source.adapter,
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

        manifest = payload.dataset_manifest_draft
        if manifest is None:
            manifest = selected_adapter.parse_to_draft_manifest(archive_path=archive_path, payload=payload)
        payload.dataset_manifest_draft = manifest

        if manifest.source.original_dataset_uuid is not None:
            await DatasetImportService.ensure_unique_source_dataset_uuid(
                project_id=project_id,
                source_dataset_uuid=manifest.source.original_dataset_uuid,
                exclude_job_id=job_id,
            )

        report = selected_adapter.validate(manifest=manifest, payload=payload)
        logger.info(
            "Validation report for job_id='{}': is_valid={}, required_user_inputs={}, blocking_errors={}",
            job_id,
            report.is_valid,
            report.required_user_inputs,
            len(report.blocking_errors),
        )
        payload.validation_report = report
        payload.step = ImportStep.WAITING_FOR_USER_INPUT

        job = await JobService.update_job_payload(
            job_id=job_id,
            payload=payload,
            status=JobStatus.PENDING,
            message="Dataset import ready for user finalization",
            progress=40,
        )
        self.queue.put((EventType.JOB_UPDATE, job))

    async def _run_commit(self, job_id: UUID, project_id: UUID, payload: DatasetImportJobPayload) -> None:
        expected_adapter_name = (
            payload.dataset_manifest_draft.source.adapter if payload.dataset_manifest_draft else None
        )
        adapter = next((a for a in self.adapters if a.__class__.__name__ == expected_adapter_name), None)
        if adapter is None:
            raise ValueError("No adapter available for commit")

        logger.info(
            "Starting commit for job_id='{}' using adapter='{}' and source='{}'",
            job_id,
            expected_adapter_name,
            payload.dataset_manifest_draft.source.source_type if payload.dataset_manifest_draft else None,
        )

        payload.step = ImportStep.IMPORTING_RESOURCE
        job = await JobService.update_job_payload(
            job_id=job_id,
            payload=payload,
            status=JobStatus.RUNNING,
            message="Starting dataset import",
            progress=60,
        )
        self.queue.put((EventType.JOB_UPDATE, job))

        dataset = await adapter.commit(payload, project_id=project_id)
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

        payload.step = ImportStep.REGISTERING_RESOURCE
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
