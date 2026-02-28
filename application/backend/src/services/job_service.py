from uuid import UUID

from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from exceptions import DuplicateJobException, ModelNotRetrainableError, ResourceNotFoundError, ResourceType
from repositories import JobRepository
from schemas import ExportJob, ImportJob, Job, TrainJob
from schemas.job import ExportJobPayload, ImportJobPayload, JobStatus, JobType, TrainJobPayload
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
    async def get_pending_train_job() -> TrainJob | ImportJob | ExportJob | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_pending_job_by_type(JobType.TRAINING)

    @staticmethod
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
            return await repo.update(job, updates)
