from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter, field_serializer

from schemas.base_job import BaseJob, JobType
from schemas.import_job import DatasetImportJobPayload

# ---------------------------------------------------------------------------
# Payload models (no ``type`` discriminator – the parent Job carries it)
# ---------------------------------------------------------------------------


class TrainJobPayload(BaseModel):
    project_id: UUID
    dataset_id: UUID
    policy: str
    model_name: str
    max_steps: int = Field(default=100, ge=100, le=100_000, description="Number of training steps")
    batch_size: int = Field(default=8, ge=1, le=256, description="Training batch size")
    num_workers: int | Literal["auto"] = Field(default="auto", description="DataLoader workers ('auto' or 0-16)")
    auto_scale_batch_size: bool = Field(
        default=False, description="Run batch-size finder before training (power scaling)"
    )
    base_model_id: UUID | None = Field(default=None, description="Model ID to resume training from")

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)

    @field_serializer("dataset_id")
    def serialize_dataset_id(self, dataset_id: UUID, _info: Any) -> str:
        return str(dataset_id)

    @field_serializer("base_model_id")
    def serialize_base_model_id(self, base_model_id: UUID | None, _info: Any) -> str | None:
        return str(base_model_id) if base_model_id else None


class ImportJobPayload(BaseModel):
    project_id: UUID
    model_name: str
    upload_file_path: str
    original_filename: str

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)


class ExportJobPayload(BaseModel):
    project_id: UUID
    model_id: UUID
    model_name: str

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)

    @field_serializer("model_id")
    def serialize_model_id(self, model_id: UUID, _info: Any) -> str:
        return str(model_id)


# ---------------------------------------------------------------------------
# Concrete Job variants (discriminated on ``type``)
# ---------------------------------------------------------------------------


class TrainJob(BaseJob):
    type: Literal[JobType.TRAINING] = JobType.TRAINING  # type: ignore[valid-type]
    payload: TrainJobPayload


class DatasetImportJob(BaseJob):
    type: Literal[JobType.DATASET_IMPORT] = JobType.DATASET_IMPORT
    payload: DatasetImportJobPayload


class ImportJob(BaseJob):
    type: Literal[JobType.IMPORT] = JobType.IMPORT
    payload: ImportJobPayload


class ExportJob(BaseJob):
    type: Literal[JobType.EXPORT] = JobType.EXPORT
    payload: ExportJobPayload


Job = Annotated[
    TrainJob | DatasetImportJob | ImportJob | ExportJob,
    Field(discriminator="type"),
]

JobPayload = TrainJobPayload | DatasetImportJobPayload | ImportJobPayload | ExportJobPayload

JOB_ADAPTER = TypeAdapter(Job)


class JobList(BaseModel):
    jobs: list[Job]
