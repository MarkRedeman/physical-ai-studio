import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from exceptions import InvalidJobStateError, ResourceNotFoundError, ResourceType
from repositories import JobRepository
from schemas import Job
from schemas.base_job import JobStatus, JobType
from schemas.dataset_import_job import DatasetImportFinalizeInput, DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob
from services.archive_safety import cleanup_staged_archive
from services.dataset_import.staging import generate_staging_id, resolve_payload_archive_path


class DatasetImportService:
    @staticmethod
    async def _load_validated_dataset_import_job(
        repo: JobRepository,
        *,
        project_id: UUID,
        job_id: UUID,
    ) -> tuple[Job, DatasetImportJobPayload]:
        job = await repo.get_by_id(job_id)
        if job is None:
            raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
        if job.project_id != project_id:
            raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
        if job.type != JobType.DATASET_IMPORT:
            raise InvalidJobStateError("Job is not a dataset import job")
        if not isinstance(job.payload, DatasetImportJobPayload):
            raise InvalidJobStateError("Dataset import job payload is invalid")

        return job, job.payload

    @staticmethod
    async def prepare_dataset_import_job(
        project_id: UUID,
        source_hint: str = "auto",
    ) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            payload = DatasetImportJobPayload(
                archive_staging_id=generate_staging_id(),
                source_hint=source_hint,
                step=ImportStep.AWAITING_UPLOAD,
            )
            job = DatasetImportJob(
                project_id=project_id,
                payload=payload,
                message="Dataset import job prepared, awaiting upload",
            )
            try:
                return await repo.save(job)
            except IntegrityError:
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=project_id)

    @staticmethod
    async def attach_dataset_import_archive(
        project_id: UUID,
        job_id: UUID,
        uploaded_archive_name: str,
    ) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job, payload = await DatasetImportService._load_validated_dataset_import_job(
                repo,
                project_id=project_id,
                job_id=job_id,
            )
            if payload.step != ImportStep.AWAITING_UPLOAD:
                raise InvalidJobStateError(
                    f"Archive can only be attached when job is in '{ImportStep.AWAITING_UPLOAD}' step"
                )

            payload.uploaded_archive_name = uploaded_archive_name
            payload.step = ImportStep.UPLOADED

            updates = {
                "payload": payload.model_dump(mode="json"),
                "status": JobStatus.PENDING,
                "message": "Dataset archive uploaded and queued",
                "progress": 5,
            }
            return await repo.update(job, updates)

    @staticmethod
    async def claim_pending_dataset_import_job() -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.claim_pending_dataset_import_job()

    @staticmethod
    async def finalize_dataset_import_job(
        project_id: UUID,
        job_id: UUID,
        finalize_input: DatasetImportFinalizeInput,
    ) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job, payload = await DatasetImportService._load_validated_dataset_import_job(
                repo,
                project_id=project_id,
                job_id=job_id,
            )
            if payload.step != ImportStep.WAITING_FOR_USER_INPUT:
                raise InvalidJobStateError(
                    f"Dataset import can only be finalized from '{ImportStep.WAITING_FOR_USER_INPUT}' step"
                )

            payload.finalize_input = finalize_input
            payload.step = ImportStep.READY_TO_COMMIT

            updates = {
                "payload": payload.model_dump(mode="json"),
                "status": JobStatus.PENDING,
                "message": "Dataset import finalized and queued",
                "progress": 45,
            }

            return await repo.update(job, updates)

    @staticmethod
    async def cancel_dataset_import_job(project_id: UUID, job_id: UUID) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job, payload = await DatasetImportService._load_validated_dataset_import_job(
                repo,
                project_id=project_id,
                job_id=job_id,
            )
            blocked_steps = {
                ImportStep.IMPORTING_RESOURCE,
                ImportStep.REGISTERING_RESOURCE,
            }
            if payload.step in blocked_steps:
                raise InvalidJobStateError("Cannot cancel import once full import has started")

            updates = {
                "status": JobStatus.CANCELED,
                "message": "Dataset import canceled",
                "end_time": datetime.datetime.now(tz=datetime.UTC),
            }
            updated_job = await repo.update(job, updates)
            cleanup_staged_archive(resolve_payload_archive_path(payload))
            return updated_job
