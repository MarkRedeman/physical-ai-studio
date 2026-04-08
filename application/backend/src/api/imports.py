from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.dependencies import get_dataset_import_job_service, get_project_id
from schemas import Job
from schemas.import_job import DatasetImportFinalizeInput, DatasetImportSource
from services.dataset_import_job_service import DatasetImportJobService
from settings import get_settings

router = APIRouter(prefix="/api/projects/{project_id}/imports", tags=["Imports"])

ProjectID = Annotated[UUID, Depends(get_project_id)]


def _persist_uploaded_archive(file: UploadFile) -> Path:
    settings = get_settings()
    cache_dir = settings.cache_dir / "imports" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or f"dataset-import-{uuid4()}.zip"
    suffix = Path(filename).suffix if Path(filename).suffix else ".zip"
    destination = cache_dir / f"{uuid4()}{suffix}"

    with destination.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            out.write(chunk)

    return destination


@router.post("/datasets", status_code=status.HTTP_202_ACCEPTED)
async def submit_dataset_import_job(
    project_id: ProjectID,
    archive: Annotated[UploadFile, File(description="Dataset archive ZIP")],
    dataset_import_job_service: Annotated[DatasetImportJobService, Depends(get_dataset_import_job_service)],
    source_hint: Annotated[str, Form()] = "auto",
) -> Job:
    """Upload a dataset archive and enqueue asynchronous import job."""
    if not archive.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing archive filename")

    if not archive.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only ZIP archives are supported"
        )

    supported_source_hints = {"auto", *[member.value for member in DatasetImportSource]}
    if source_hint not in supported_source_hints:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid source_hint '{source_hint}'. Expected one of: {sorted(supported_source_hints)}",
        )

    uploaded_archive_path = _persist_uploaded_archive(archive)
    return await dataset_import_job_service.submit_dataset_import_job(
        project_id=project_id,
        uploaded_archive_path=str(uploaded_archive_path),
        source_hint=source_hint,
    )


@router.post("/datasets/{job_id}:finalize", status_code=status.HTTP_202_ACCEPTED)
async def finalize_dataset_import_job(
    project_id: ProjectID,
    job_id: UUID,
    payload: DatasetImportFinalizeInput,
    dataset_import_job_service: Annotated[DatasetImportJobService, Depends(get_dataset_import_job_service)],
) -> Job:
    """Finalize staged dataset import by providing required user inputs."""
    return await dataset_import_job_service.finalize_dataset_import_job(
        project_id=project_id,
        job_id=job_id,
        finalize_input=payload,
    )


@router.post("/datasets/{job_id}:cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_dataset_import_job(
    project_id: ProjectID,
    job_id: UUID,
    dataset_import_job_service: Annotated[DatasetImportJobService, Depends(get_dataset_import_job_service)],
) -> Job:
    """Cancel dataset import before full import starts."""
    return await dataset_import_job_service.cancel_dataset_import_job(project_id=project_id, job_id=job_id)
