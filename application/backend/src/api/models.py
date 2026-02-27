import io
import json
import yaml
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from starlette import status
from starlette.background import BackgroundTask
from fastapi.responses import StreamingResponse

from api.dependencies import get_dataset_service, get_model_download_service, get_model_import_service, get_snapshot_service, get_model_id, get_model_service, validate_uuid
from api.utils import safe_archive_name
from api.dependencies import get_dataset_service, get_model_id, get_model_service, get_snapshot_service
from exceptions import ResourceNotFoundError, ResourceType
from internal_datasets.utils import get_internal_dataset
from schemas import Model
from services import DatasetService, ModelImportService, ModelDownloadService, ModelService, SnapshotService

from settings import get_settings

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


@router.get("/{model_id}:export")
async def export_model(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> StreamingResponse:
    """Export a model as a zip archive containing the model's export artifacts.

    The zip contains the exports/ subtree of the model directory plus a
    ``manifest.json`` with model metadata.  The archive is structured as
    ``{model_name}/exports/{backend}/{files}`` so that a future import
    endpoint can extract it directly into a new model directory.
    """
    model = await model_service.get_model_by_id(model_id)
    model_path = Path(model.path).expanduser()
    exports_path = model_path / "exports"

    if not exports_path.is_dir():
        raise ResourceNotFoundError(
            ResourceType.MODEL,
            str(model_id),
            message=f"Model exports not found for model '{model.name}'. The model may not have been exported yet.",
        )

    settings = get_settings()
    manifest = {
        "name": model.name,
        "policy": model.policy,
        "properties": model.properties,
        "original_model_id": str(model.id),
        "original_project_id": str(model.project_id),
        "original_dataset_id": str(model.dataset_id) if model.dataset_id else None,
        "original_snapshot_id": str(model.snapshot_id) if model.snapshot_id else None,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "physicalai_version": settings.version,
    }

    def _generate_zip():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add manifest
            zf.writestr(
                str(Path(model.name) / "manifest.json"),
                json.dumps(manifest, indent=2),
            )
            # Add export artifacts
            for file_path in sorted(exports_path.rglob("*")):
                if file_path.is_file():
                    arcname = str(Path(model.name) / file_path.relative_to(model_path))
                    zf.write(file_path, arcname)
        buffer.seek(0)
        yield buffer.read()

    # Sanitize the model name for use in the filename
    safe_name = model.name.replace('"', "").replace("/", "_").replace("\\", "_")
    filename = f"{safe_name}.zip"

    return StreamingResponse(
        _generate_zip(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(":import", status_code=status.HTTP_201_CREATED)
async def import_model(
    file: UploadFile,
    project_id: Annotated[str, Form()],
    name: Annotated[str, Form()],
    model_import_service: Annotated[ModelImportService, Depends(get_model_import_service)],
) -> Model:
    """Import a model from a zip archive.

    Supports two archive formats:

    **Physical AI Studio export** — a zip containing ``exports/{backend}/metadata.yaml``
    and optionally a ``manifest.json`` for reconnecting dataset/snapshot relationships.

    **HuggingFace/LeRobot model** — a zip containing ``config.json`` and
    ``model.safetensors``.  The model is converted to the Physical AI Studio
    inference format (``exports/torch/{policy}.pt + metadata.yaml``) on import.
    """
    contents = await file.read()
    return await model_import_service.import_model(
        file_content=contents,
        filename=file.filename or "",
        name=name,
        project_id=project_id,
    )


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
