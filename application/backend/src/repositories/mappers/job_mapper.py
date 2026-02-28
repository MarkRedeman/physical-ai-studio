from pydantic import TypeAdapter

from db.schema import JobDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.job import ExportJob, ImportJob, Job, TrainJob

# TypeAdapter handles the Annotated discriminated union properly.
_job_adapter: TypeAdapter[TrainJob | ImportJob | ExportJob] = TypeAdapter(Job)


class JobMapper(IBaseMapper):
    @staticmethod
    def to_schema(job: TrainJob | ImportJob | ExportJob) -> JobDB:
        data = job.model_dump()
        # Flatten typed payload back to a plain dict for DB storage.
        data["payload"] = job.payload.model_dump(mode="json")
        return JobDB(**data)

    @staticmethod
    def from_schema(job_db: JobDB) -> TrainJob | ImportJob | ExportJob:
        data = {c.key: getattr(job_db, c.key) for c in JobDB.__table__.columns}
        # Inject ``type`` into the dict so the discriminated union can resolve.
        return _job_adapter.validate_python(data)
