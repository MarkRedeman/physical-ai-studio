import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from db.schema import JobDB
from exceptions import (
    DuplicateImportSourceError,
    DuplicateJobException,
    InvalidJobStateError,
    ModelNotRetrainableError,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
)
from repositories import JobRepository
from schemas import Job
from schemas.base_job import JobStatus, JobType
from schemas.import_job import DatasetImportFinalizeInput, DatasetImportJobPayload, ImportStep
from schemas.job import (
    DatasetImportJob,
    ExportJob,
    ExportJobPayload,
    ImportJob,
    ImportJobPayload,
    JobPayload,
    TrainJob,
    TrainJobPayload,
)
from services.environment_service import EnvironmentService
from services.model_service import ModelService


class JobService:
    @staticmethod
    async def get_job_list(extra_filters: dict | None = None) -> list[TrainJob | ImportJob | ExportJob]:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_all(extra_filters=extra_filters)

    @staticmethod
    async def get_job_by_id(job_id: UUID) -> TrainJob | ImportJob | ExportJob:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
            return job

    @staticmethod
    async def get_jobs_by_ids(job_ids: list[UUID]) -> list[Job]:
        """Fetch multiple jobs by id in a single query."""
        if not job_ids:
            return []

        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_all(expressions=[JobDB.id.in_([str(job_id) for job_id in job_ids])])

    @staticmethod
    async def submit_train_job(payload: TrainJobPayload) -> TrainJob:
        # Validate that the base model (if any) supports retraining.
        if payload.base_model_id is not None:
            base_model = await ModelService.get_model_by_id(payload.base_model_id)
            if base_model.properties.get("source") == "huggingface":
                raise ModelNotRetrainableError(base_model.name)

        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            if await repo.is_job_duplicate(project_id=payload.project_id, payload=payload):
                raise DuplicateJobException

            try:
                job = TrainJob(
                    project_id=payload.project_id,
                    payload=payload,
                    message="Training job submitted",
                )
                return await repo.save(job)
            except IntegrityError:
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=payload.project_id)

    @staticmethod
    async def submit_dataset_import_job(
        project_id: UUID,
        uploaded_archive_path: str,
        source_hint: str = "auto",
    ) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            payload = DatasetImportJobPayload(
                uploaded_archive_path=uploaded_archive_path,
                source_hint=source_hint,
                step=ImportStep.UPLOADED,
            )
            job = DatasetImportJob(
                project_id=project_id,
                payload=payload,
                message="Dataset import submitted",
            )
            try:
                return await repo.save(job)
            except IntegrityError:
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=project_id)

    @staticmethod
    async def get_pending_train_job() -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_pending_job_by_type(JobType.TRAINING)

    @staticmethod
    async def get_pending_dataset_import_job() -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_pending_dataset_import_job()

    @staticmethod
    async def update_job_payload(
        job_id: UUID,
        payload: JobPayload,
        *,
        status: JobStatus | None = None,
        message: str | None = None,
        progress: int | None = None,
        extra_info: dict | None = None,
    ) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundError(ResourceType.JOB, resource_id=job_id)

            updates: dict = {"payload": payload.model_dump(mode="json")}
            if status is not None:
                updates["status"] = status
            if message is not None:
                updates["message"] = message
            if progress is not None:
                updates["progress"] = progress
            if extra_info is not None:
                updates["extra_info"] = extra_info
            if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED}:
                updates["end_time"] = datetime.datetime.now(tz=datetime.UTC)

            return await repo.update(job, updates)

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

    async def submit_import_job(payload: ImportJobPayload) -> ImportJob:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            try:
                job = ImportJob(
                    project_id=payload.project_id,
                    payload=payload,
                    message="Import job submitted",
                )
                return await repo.save(job)
            except IntegrityError:
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=payload.project_id)

    @staticmethod
    async def submit_export_job(payload: ExportJobPayload) -> ExportJob:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            try:
                job = ExportJob(
                    project_id=payload.project_id,
                    payload=payload,
                    message="Export job submitted",
                )
                return await repo.save(job)
            except IntegrityError:
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=payload.project_id)

    @staticmethod
    async def get_pending_import_export_job() -> TrainJob | ImportJob | ExportJob | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_pending_job_by_types([JobType.IMPORT, JobType.EXPORT])

    @staticmethod
    async def update_job_status(
        job_id: UUID,
        status: JobStatus,
        message: str | None = None,
        progress: int | None = None,
        extra_info: dict | None = None,
    ) -> TrainJob | ImportJob | ExportJob:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundError(ResourceType.JOB, resource_id=job_id)
            updates: dict = {"status": status}
            if message is not None:
                updates["message"] = message
            progress_ = 100 if status is JobStatus.COMPLETED else progress
            if progress_ is not None:
                updates["progress"] = progress_
            if extra_info is not None:
                updates["extra_info"] = extra_info
            if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED}:
                updates["end_time"] = datetime.datetime.now(tz=datetime.UTC)
            return await repo.update(job, updates)

    @staticmethod
    async def delete_job(job_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job: Job | None = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundError(ResourceType.JOB, str(job_id))

            if job.status not in {JobStatus.FAILED, JobStatus.CANCELED}:
                raise ResourceInUseError(ResourceType.JOB, str(job_id))

            await repo.delete_by_id(job_id)
