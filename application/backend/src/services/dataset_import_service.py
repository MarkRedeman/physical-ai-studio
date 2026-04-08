import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from exceptions import DuplicateImportSourceError, InvalidJobStateError, ResourceNotFoundError, ResourceType
from repositories import JobRepository
from schemas import Job
from schemas.base_job import JobStatus, JobType
from schemas.import_job import DatasetImportFinalizeInput, DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob
from services.environment_service import EnvironmentService


class DatasetImportService:
    @staticmethod
    async def prepare_dataset_import_job(
        project_id: UUID,
        source_hint: str = "auto",
    ) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            payload = DatasetImportJobPayload(
                uploaded_archive_path="",
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
        uploaded_archive_path: str,
    ) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
            if str(job.project_id) != str(project_id):
                raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
            if job.type != JobType.DATASET_IMPORT:
                raise InvalidJobStateError("Job is not a dataset import job")
            if not isinstance(job.payload, DatasetImportJobPayload):
                raise InvalidJobStateError("Dataset import job payload is invalid")

            payload = job.payload
            if payload.step != ImportStep.AWAITING_UPLOAD:
                raise InvalidJobStateError(
                    f"Archive can only be attached when job is in '{ImportStep.AWAITING_UPLOAD}' step"
                )

            payload.uploaded_archive_path = uploaded_archive_path
            payload.step = ImportStep.UPLOADED

            updates = {
                "payload": payload.model_dump(mode="json"),
                "status": JobStatus.PENDING,
                "message": "Dataset archive uploaded and queued",
                "progress": 5,
            }
            return await repo.update(job, updates)

    @staticmethod
    async def get_pending_dataset_import_job() -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_pending_dataset_import_job()

    @staticmethod
    async def finalize_dataset_import_job(
        project_id: UUID,
        job_id: UUID,
        finalize_input: DatasetImportFinalizeInput,
    ) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
            if str(job.project_id) != str(project_id):
                raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
            if job.type != JobType.DATASET_IMPORT:
                raise InvalidJobStateError("Job is not a dataset import job")

            if not isinstance(job.payload, DatasetImportJobPayload):
                raise InvalidJobStateError("Dataset import job payload is invalid")
            payload = job.payload
            if payload.step != ImportStep.WAITING_FOR_USER_INPUT:
                raise InvalidJobStateError(
                    f"Dataset import can only be finalized from '{ImportStep.WAITING_FOR_USER_INPUT}' step"
                )

            await EnvironmentService.get_environment_by_id(
                project_id=project_id, environment_id=finalize_input.environment_id
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
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
            if str(job.project_id) != str(project_id):
                raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
            if job.type != JobType.DATASET_IMPORT:
                raise InvalidJobStateError("Job is not a dataset import job")

            if not isinstance(job.payload, DatasetImportJobPayload):
                raise InvalidJobStateError("Dataset import job payload is invalid")
            payload = job.payload
            blocked_steps = {
                ImportStep.IMPORTING_RESOURCE,
                ImportStep.EXTRACTING_ARCHIVE,
                ImportStep.REGISTERING_RESOURCE,
            }
            if payload.step in blocked_steps:
                raise InvalidJobStateError("Cannot cancel import once full import has started")

            updates = {
                "status": JobStatus.CANCELED,
                "message": "Dataset import canceled",
                "end_time": datetime.datetime.now(tz=datetime.UTC),
            }
            return await repo.update(job, updates)

    @staticmethod
    async def ensure_unique_source_dataset_uuid(
        project_id: UUID,
        source_dataset_uuid: UUID,
        *,
        exclude_job_id: UUID | None = None,
    ) -> None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            jobs = await repo.get_jobs_by_type(project_id=project_id, job_type=JobType.DATASET_IMPORT)
            source_dataset_uuid_str = str(source_dataset_uuid)

            for job in jobs:
                if exclude_job_id is not None and str(job.id) == str(exclude_job_id):
                    continue
                if not isinstance(job.payload, DatasetImportJobPayload):
                    continue
                payload = job.payload
                source_uuid = (
                    payload.dataset_manifest_draft.source.original_dataset_uuid
                    if payload.dataset_manifest_draft is not None
                    else None
                )
                if source_uuid and str(source_uuid) == source_dataset_uuid_str:
                    raise DuplicateImportSourceError(resource_kind="Dataset", source_uuid=source_dataset_uuid_str)
