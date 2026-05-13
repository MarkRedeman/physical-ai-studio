import asyncio
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.dependencies import get_model_import_service, get_project_id
from schemas import  Model
from services.model_import_service import ModelImportService

router = APIRouter(prefix="/api/projects/{project_id}/imports", tags=["Imports"])

ProjectID = Annotated[UUID, Depends(get_project_id)]

_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def _write_archive_to_disk(file: UploadFile, destination: Path) -> None:
    """Synchronous helper: write upload to *destination* in chunks."""
    with destination.open("wb") as out:
        while chunk := file.file.read(_UPLOAD_CHUNK_SIZE):
            out.write(chunk)


@router.post("/models:import", status_code=status.HTTP_201_CREATED)
async def import_model_archive(  # noqa: PLR0913
    model_import_service: Annotated[ModelImportService, Depends(get_model_import_service)],
    project_id: ProjectID,
    archive: Annotated[UploadFile, File(description="Model archive ZIP")],
    dataset_id: Annotated[UUID, Form()],
    policy: Annotated[str, Form()],
    model_name: Annotated[str, Form()],
    base_model_id: Annotated[UUID | None, Form()] = None,
    version: Annotated[int, Form()] = 1,
) -> Model:
    """Import a model archive in a single request without staging steps."""
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
            policy=policy,
            model_name=model_name,
            base_model_id=base_model_id,
            version=version,
        )
    finally:
        temp_archive_path.unlink(missing_ok=True)
