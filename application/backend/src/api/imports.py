import asyncio
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.dependencies import get_dataset_import_service, get_project_id
from schemas import Job
from schemas.import_job import DatasetImportFinalizeInput, DatasetImportSource
from services.dataset_import_service import DatasetImportService
from settings import get_settings

router = APIRouter(prefix="/api/projects/{project_id}/imports", tags=["Imports"])

ProjectID = Annotated[UUID, Depends(get_project_id)]

_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def _get_upload_cache_dir() -> Path:
    settings = get_settings()
    cache_dir = settings.cache_dir / "imports" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _write_archive_to_disk(file: UploadFile, destination: Path) -> None:
    """Synchronous helper: write upload to *destination* in chunks."""
    with destination.open("wb") as out:
        while chunk := file.file.read(_UPLOAD_CHUNK_SIZE):
            out.write(chunk)


async def _persist_uploaded_archive(file: UploadFile) -> Path:
    """Persist an uploaded archive to a temp location without blocking the event loop."""
    cache_dir = _get_upload_cache_dir()
    filename = file.filename or f"dataset-import-{uuid4()}.zip"
    suffix = Path(filename).suffix if Path(filename).suffix else ".zip"
    destination = cache_dir / f"{uuid4()}{suffix}"
    await asyncio.to_thread(_write_archive_to_disk, file, destination)
    return destination


@router.post("/datasets:prepare", status_code=status.HTTP_202_ACCEPTED)
async def prepare_dataset_import_job(
    project_id: ProjectID,
    job_service: Annotated[DatasetImportService, Depends(get_dataset_import_service)],
    source_hint: Annotated[str, Form()] = "auto",
) -> Job:
    """Phase 1: Create a dataset import job immediately, before the archive is uploaded.

    Returns a Job with step=awaiting_upload. Use the returned job_id in the phase-2
    upload endpoint to attach the archive and queue processing.
    """
    supported_source_hints = {"auto", *[member.value for member in DatasetImportSource]}
    if source_hint not in supported_source_hints:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid source_hint '{source_hint}'. Expected one of: {sorted(supported_source_hints)}",
        )

    return await job_service.prepare_dataset_import_job(project_id=project_id, source_hint=source_hint)


@router.put("/datasets/{job_id}:upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_dataset_import_archive(
    project_id: ProjectID,
    job_id: UUID,
    archive: Annotated[UploadFile, File(description="Dataset archive ZIP")],
    job_service: Annotated[DatasetImportService, Depends(get_dataset_import_service)],
) -> Job:
    """Phase 2: Upload the archive and attach it to an existing import job.

    The job must be in the awaiting_upload step (created via POST /datasets:prepare).
    After this call the job transitions to step=uploaded and status=pending so the
    worker can pick it up for processing.
    """
    if not archive.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing archive filename")

    if not archive.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only ZIP archives are supported"
        )

    uploaded_archive_path = await _persist_uploaded_archive(archive)
    return await job_service.attach_dataset_import_archive(
        project_id=project_id,
        job_id=job_id,
        uploaded_archive_path=str(uploaded_archive_path),
    )


@router.post("/datasets/{job_id}:finalize", status_code=status.HTTP_202_ACCEPTED)
async def finalize_dataset_import_job(
    project_id: ProjectID,
    job_id: UUID,
    payload: DatasetImportFinalizeInput,
    job_service: Annotated[DatasetImportService, Depends(get_dataset_import_service)],
) -> Job:
    """Finalize staged dataset import by providing required user inputs."""
    return await job_service.finalize_dataset_import_job(
        project_id=project_id,
        job_id=job_id,
        finalize_input=payload,
    )


@router.post("/datasets/{job_id}:cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_dataset_import_job(
    project_id: ProjectID,
    job_id: UUID,
    job_service: Annotated[DatasetImportService, Depends(get_dataset_import_service)],
) -> Job:
    """Cancel dataset import before full import starts."""
    return await job_service.cancel_dataset_import_job(project_id=project_id, job_id=job_id)
