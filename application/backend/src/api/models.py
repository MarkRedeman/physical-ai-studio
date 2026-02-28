import logging
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from starlette import status
from starlette.background import BackgroundTask

from api.dependencies import (
    get_dataset_service,
    get_job_service,
    get_model_download_service,
    get_model_id,
    get_model_service,
)
from api.utils import safe_archive_name
from exceptions import ResourceNotFoundError, ResourceType
from internal_datasets.utils import get_internal_dataset
from schemas import  Model
from services import DatasetService, JobService, ModelDownloadService, ModelService
from schemas.job import ExportJob, ExportJobPayload, ImportJob, ImportJobPayload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("/{model_id}")
async def get_model_by_id(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> Model:
    """Get model by id."""
    return await model_service.get_model_by_id(model_id)


@router.get("/{model_id}/tasks")
async def get_tasks_of_model(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[str]:
    """Get availabe tasks for model."""
    model = await model_service.get_model_by_id(model_id)
    if model.dataset_id is None:
        return []
    dataset = await dataset_service.get_dataset_by_id(model.dataset_id)
    return get_internal_dataset(dataset).get_tasks()


@router.get("/{model_id}/download")
async def model_download_endpoint(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    model_download_service: Annotated[ModelDownloadService, Depends(get_model_download_service)],
    include_snapshot: bool = False,
) -> FileResponse:
    """Download model folder as a zip archive.

    By default the dataset snapshot that was used for training is excluded
    from the archive.  Pass ``include_snapshot=true`` to include it.
    """
    model = await model_service.get_model_by_id(model_id)
    model_path = Path(model.path).resolve()

    if not model_path.exists() or not model_path.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model path not found.")

    archive_path = model_download_service.create_model_archive(model_path, include_snapshot=include_snapshot)
    filename = f"{safe_archive_name(model.name, fallback='model')}.zip"
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@router.post("/{model_id}:export")
async def export_model(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> ExportJob:
    """Submit an export job for a model.

    The export runs as a background job. When completed, the zip archive
    can be downloaded via GET /api/models/{model_id}/export/download.
    """
    model = await model_service.get_model_by_id(model_id)
    exports_path = Path(model.path).expanduser() / "exports"

    if not exports_path.is_dir():
        raise ResourceNotFoundError(
            ResourceType.MODEL,
            str(model_id),
            message=f"Model exports not found for model '{model.name}'. The model may not have been exported yet.",
        )

    payload = ExportJobPayload(
        project_id=model.project_id,
        model_id=model.id,
        model_name=model.name,
    )
    return await job_service.submit_export_job(payload)


@router.get("/{model_id}/export/download")
async def download_export(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> FileResponse:
    """Download a previously exported model zip archive."""
    model = await model_service.get_model_by_id(model_id)
    model_path = Path(model.path).expanduser()
    archive_dir = model_path / "exports" / "archive"

    if not archive_dir.is_dir():
        raise ResourceNotFoundError(
            ResourceType.MODEL,
            str(model_id),
            message="No export archive found. Please run an export job first.",
        )

    # Find the zip file in the archive directory
    zip_files = list(archive_dir.glob("*.zip"))
    if not zip_files:
        raise ResourceNotFoundError(
            ResourceType.MODEL,
            str(model_id),
            message="No export archive found. Please run an export job first.",
        )

    zip_path = zip_files[0]
    safe_name = model.name.replace('"', "").replace("/", "_").replace("\\", "_")
    filename = f"{safe_name}.zip"

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=filename,
    )


@router.post(":import", status_code=status.HTTP_202_ACCEPTED)
async def import_model(
    file: UploadFile,
    project_id: Annotated[str, Form()],
    name: Annotated[str, Form()],
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> ImportJob:
    """Submit a model import job.

    The uploaded file is saved to a temporary location and processed
    by the ImportExportWorker in the background.
    """
    # Save uploaded file to temp location
    contents = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.write(contents)
    tmp.close()

    payload = ImportJobPayload(
        project_id=UUID(project_id),
        model_name=name,
        upload_file_path=tmp.name,
        original_filename=file.filename or "upload.zip",
    )
    return await job_service.submit_import_job(payload)


@router.delete("/{model_id}")
async def remove_model(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> None:
    """Fetch all projects."""
    model = await model_service.get_model_by_id(model_id)
    if model is None:
        raise ResourceNotFoundError(ResourceType.MODEL, model_id)
    await model_service.delete_model(model)
