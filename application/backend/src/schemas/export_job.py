"""Schema for model re-export jobs."""

from __future__ import annotations

from typing import Literal
from uuid import UUID  # noqa: TC003 - pydantic needs it at runtime to build the schema

from pydantic import BaseModel, Field

from schemas.base_job import BaseJob, JobType


class ModelExportJobPayload(BaseModel):
    """What a model re-export job should produce."""

    model_id: UUID = Field(..., description="Source model to re-export")
    backends: list[Literal["torch", "openvino", "onnx", "executorch"]] = Field(
        default=["torch", "openvino"],
        description="Backends to export to",
    )
    compress: bool = Field(default=True, description="Apply NNCF INT8 compression to the OpenVINO export")
    name: str | None = Field(default=None, description="Optional display name for the exported model")


class ModelExportJob(BaseJob):
    type: Literal[JobType.MODEL_EXPORT] = JobType.MODEL_EXPORT  # type: ignore[valid-type]
    payload: ModelExportJobPayload
