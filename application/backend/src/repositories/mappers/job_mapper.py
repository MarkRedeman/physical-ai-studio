from pydantic import TypeAdapter

from db.schema import JobDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.job import Job

# TypeAdapter handles the Annotated discriminated union properly.
_job_adapter: TypeAdapter[Job] = TypeAdapter(Job)

JOB_ADAPTER = TypeAdapter(Job)


class JobMapper(IBaseMapper):
    @staticmethod
    def to_schema(db_schema: Job) -> JobDB:
        data = db_schema.model_dump()
        # Flatten typed payload back to a plain dict for DB storage.
        data["payload"] = db_schema.payload.model_dump(mode="json")
        return JobDB(**data)

    @staticmethod
    def from_schema(model: JobDB) -> Job:
        return JOB_ADAPTER.validate_python(model, from_attributes=True)
