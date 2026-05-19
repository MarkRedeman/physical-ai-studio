import asyncio
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from sse_starlette import EventSourceResponse
from starlette import status
from starlette.background import BackgroundTask

from api.dependencies import (
    get_dataset_service,
    get_model_download_service,
    get_model_id,
    get_model_import_service,
    get_model_metrics_service,
    get_model_service,
)
from api.utils import safe_archive_name
from exceptions import ResourceNotFoundError, ResourceType
from internal_datasets.utils import get_internal_dataset
from schemas import Model
from services import DatasetService, ModelDownloadService, ModelImportService, ModelMetricsService, ModelService

_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024


def _write_archive_to_disk(file: UploadFile, destination: Path) -> None:
    with destination.open("wb") as out:
        while chunk := file.file.read(_UPLOAD_CHUNK_SIZE):
            out.write(chunk)


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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model has no dataset associated.")
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


@router.get("/{model_id}/metrics")
async def stream_metrics(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    model_metrics_service: Annotated[ModelMetricsService, Depends(get_model_metrics_service)],
) -> EventSourceResponse:
    """Get an EventSourceResponse from the metrics of a model."""
    model = await model_service.get_model_by_id(model_id)
    metrics_path = await model_metrics_service.get_model_metrics_path(model)
    if metrics_path.exists():
        return EventSourceResponse(model_metrics_service.tail_csv_file(metrics_path))
    return EventSourceResponse(model_metrics_service.empty_metrics_stream())


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


@router.post(":import", status_code=status.HTTP_201_CREATED)
async def import_model(
    archive: Annotated[UploadFile, File(description="Model archive ZIP")],
    project_id: Annotated[UUID, Form()],
    dataset_id: Annotated[UUID, Form()],
    model_name: Annotated[str, Form()],
    model_import_service: Annotated[ModelImportService, Depends(get_model_import_service)],
    base_model_id: Annotated[UUID | None, Form()] = None,
    version: Annotated[int, Form()] = 1,
) -> Model:
    """Import a model ZIP produced by the model download endpoint."""
    if not archive.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing archive filename")

    if not archive.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only ZIP archives are supported",
        )

    temp_archive_path = Path(tempfile.gettempdir()) / f"model-import-{uuid4()}.zip"

    try:
        await asyncio.to_thread(_write_archive_to_disk, archive, temp_archive_path)
        return await model_import_service.import_model_archive(
            archive_path=temp_archive_path,
            project_id=project_id,
            dataset_id=dataset_id,
            model_name=model_name,
            base_model_id=base_model_id,
            version=version,
        )
    finally:
        temp_archive_path.unlink(missing_ok=True)
