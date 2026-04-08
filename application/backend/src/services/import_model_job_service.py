from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories import JobRepository
from schemas.job import ImportJob, ImportJobPayload


class ImportModelJobService:
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
