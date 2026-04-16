from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pydantic_core.core_schema import SerializationInfo

from schemas.base_job import JobType


class ManifestCameraEntry(BaseModel):
    """Recording schema entry describing a single camera stream."""

    name: str
    width: int | None = None
    height: int | None = None
    fps: int | None = None


class ManifestRobotEntry(BaseModel):
    """Recording schema entry describing a robot and its controllable joints."""

    name: str
    type: str | None = None
    joints: list[str] = Field(default_factory=list)


class DatasetManifestRecordingSchema(BaseModel):
    """Cameras and robots inferred from dataset source metadata."""

    cameras: list[ManifestCameraEntry] = Field(default_factory=list)
    robots: list[ManifestRobotEntry] = Field(default_factory=list)


class ImportStep(StrEnum):
    AWAITING_UPLOAD = "awaiting_upload"
    UPLOADED = "uploaded"
    DETECTING_SOURCE = "detecting_source"
    GENERATING_DRAFT_MANIFEST = "generating_draft_manifest"
    WAITING_FOR_USER_INPUT = "waiting_for_user_input"
    READY_TO_COMMIT = "ready_to_commit"
    REGISTERING_RESOURCE = "registering_resource"
    IMPORTING_RESOURCE = "importing_resource"
    COMPLETED = "completed"


class DatasetImportSource(StrEnum):
    LEROBOT_V2 = "lerobot_v2"
    LEROBOT_V3 = "lerobot_v3"
    UNKNOWN = "unknown"


class ImportValidationIssue(BaseModel):
    code: str
    message: str
    path: str | None = None


class ImportValidationReport(BaseModel):
    is_valid: bool = True
    blocking_errors: list[ImportValidationIssue] = Field(default_factory=list)
    warnings: list[ImportValidationIssue] = Field(default_factory=list)
    required_user_inputs: list[str] = Field(default_factory=list)


class DatasetManifestCapture(BaseModel):
    fps: int | None = None
    episode_count: int | None = None
    frame_count: int | None = None


class DatasetManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_type: DatasetImportSource = DatasetImportSource.UNKNOWN
    suggested_name: str | None = None
    capture: DatasetManifestCapture = Field(default_factory=DatasetManifestCapture)
    schema_: DatasetManifestRecordingSchema = Field(default_factory=DatasetManifestRecordingSchema, alias="schema")


class DatasetImportFinalizeInput(BaseModel):
    dataset_name: str
    environment_id: UUID
    default_task: str | None = None

    @field_serializer("environment_id")
    def serialize_environment_id(self, environment_id: UUID, _info: SerializationInfo) -> str:
        return str(environment_id)


class DatasetImportJobPayload(BaseModel):
    type: Literal[JobType.DATASET_IMPORT] = JobType.DATASET_IMPORT
    step: ImportStep = ImportStep.AWAITING_UPLOAD
    result_dataset_id: UUID | None = None

    # Opaque staging identifier - resolve the archive path via staging_path_for_id().
    archive_staging_id: str | None = None
    uploaded_archive_name: str | None = None
    source_hint: str = "auto"
    dataset_manifest_draft: DatasetManifest | None = None
    validation_report: ImportValidationReport | None = None
    finalize_input: DatasetImportFinalizeInput | None = None

    @field_validator("archive_staging_id", mode="before")
    @classmethod
    def validate_archive_staging_id(cls, v: object) -> str | None:
        """Accept None or a canonical UUID string; reject anything else."""
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("archive_staging_id must be a string or None")
        try:
            UUID(v)
        except ValueError:
            raise ValueError(f"archive_staging_id must be a valid UUID string, got: {v!r}")
        return v

    @field_serializer("result_dataset_id")
    def serialize_result_dataset_id(self, result_dataset_id: UUID | None, _info: SerializationInfo) -> str | None:
        return str(result_dataset_id) if result_dataset_id else None
