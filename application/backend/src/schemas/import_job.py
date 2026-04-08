from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer

from schemas.base_job import JobType


class ImportStep(StrEnum):
    AWAITING_UPLOAD = "awaiting_upload"
    UPLOADED = "uploaded"
    DETECTING_SOURCE = "detecting_source"
    GENERATING_DRAFT_MANIFEST = "generating_draft_manifest"
    WAITING_FOR_USER_INPUT = "waiting_for_user_input"
    VALIDATING_FINAL_INPUT = "validating_final_input"
    READY_TO_COMMIT = "ready_to_commit"
    EXTRACTING_ARCHIVE = "extracting_archive"
    REGISTERING_RESOURCE = "registering_resource"
    IMPORTING_RESOURCE = "importing_resource"


class DatasetImportSource(StrEnum):
    STUDIO = "studio"
    LEROBOT_V2 = "lerobot_v2"
    LEROBOT_V3 = "lerobot_v3"
    TROSSEN_SDK = "trossen_sdk"
    UNKNOWN = "unknown"


class ModelImportSource(StrEnum):
    STUDIO = "studio"
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
    compatibility_summary: dict = Field(default_factory=dict)


class DatasetManifestSource(BaseModel):
    source_type: DatasetImportSource = DatasetImportSource.UNKNOWN
    adapter: str | None = None
    source_format_version: str | None = None
    source_identifier: str | None = None
    original_dataset_uuid: UUID | None = None

    @field_serializer("original_dataset_uuid")
    def serialize_original_dataset_uuid(self, original_dataset_uuid: UUID | None) -> str | None:
        return str(original_dataset_uuid) if original_dataset_uuid else None


class DatasetManifestIdentity(BaseModel):
    suggested_name: str | None = None
    default_task: str | None = None


class DatasetManifestCapture(BaseModel):
    fps: int | None = None
    episode_count: int | None = None
    frame_count: int | None = None


class DatasetManifestV1(BaseModel):
    manifest_version: str = "1.0"
    resource_type: Literal["dataset"] = "dataset"
    source: DatasetManifestSource = Field(default_factory=DatasetManifestSource)
    identity: DatasetManifestIdentity = Field(default_factory=DatasetManifestIdentity)
    capture: DatasetManifestCapture = Field(default_factory=DatasetManifestCapture)
    schema: dict = Field(default_factory=dict)
    integrity: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class DatasetImportFinalizeInput(BaseModel):
    dataset_name: str
    environment_id: UUID
    default_task: str | None = None
    user_overrides: dict = Field(default_factory=dict)

    @field_serializer("environment_id")
    def serialize_environment_id(self, environment_id: UUID) -> str:
        return str(environment_id)


class DatasetImportJobPayload(BaseModel):
    type: Literal[JobType.DATASET_IMPORT] = JobType.DATASET_IMPORT
    step: ImportStep = ImportStep.UPLOADED
    result_dataset_id: UUID | None = None

    uploaded_archive_path: str
    source_hint: DatasetImportSource | Literal["auto"] = "auto"
    dataset_manifest_draft: DatasetManifestV1 | None = None
    validation_report: ImportValidationReport | None = None
    finalize_input: DatasetImportFinalizeInput | None = None

    @field_serializer("result_dataset_id")
    def serialize_result_dataset_id(self, result_dataset_id: UUID | None) -> str | None:
        return str(result_dataset_id) if result_dataset_id else None


class ModelImportJobPayload(BaseModel):
    type: Literal[JobType.MODEL_IMPORT] = JobType.MODEL_IMPORT
    project_id: UUID
    step: ImportStep = ImportStep.UPLOADED
    uploaded_archive_path: str
    uploaded_archive_name: str
    source_hint: ModelImportSource | Literal["auto"] = "auto"

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID) -> str:
        return str(project_id)


class ModelImportFinalizeInput(BaseModel):
    model_name: str
    policy: str
    linking: dict = Field(default_factory=dict)
    user_overrides: dict = Field(default_factory=dict)
