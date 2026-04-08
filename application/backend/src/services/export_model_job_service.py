from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories import JobRepository
from schemas.job import ExportJob, ExportJobPayload


class ExportModelJobService:
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
