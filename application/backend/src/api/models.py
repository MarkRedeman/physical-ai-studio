import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import UUID

import yaml
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_dataset_service, get_model_id, get_model_service, get_snapshot_service
from exceptions import ResourceNotFoundError, ResourceType
from internal_datasets.utils import get_internal_dataset
from schemas import Model
from services import DatasetService, ModelService, SnapshotService
from settings import get_settings

router = APIRouter(prefix="/api/models", tags=["Models"])

# Known policy class paths mapped to short policy names
_POLICY_CLASS_MAP: dict[str, str] = {
    "physicalai.policies.act.policy.ACT": "act",
    "physicalai.policies.pi0.policy.Pi0": "pi0",
    "physicalai.policies.smolvla.policy.SmolVLA": "smolvla",
}


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


@router.get("/{model_id}/export")
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


def _extract_policy_from_metadata(metadata: dict) -> str:
    """Extract short policy name from metadata.yaml contents."""
    policy_class = metadata.get("policy_class", "")
    if policy_class in _POLICY_CLASS_MAP:
        return _POLICY_CLASS_MAP[policy_class]
    # Fallback: use the last segment of the class path, lowercased
    parts = policy_class.rsplit(".", 1)
    return parts[-1].lower() if parts else "unknown"


@router.post(":import", status_code=status.HTTP_201_CREATED)
async def import_model(
    file: UploadFile,
    project_id: Annotated[str, Form()],
    name: Annotated[str, Form()],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    snapshot_service: Annotated[SnapshotService, Depends(get_snapshot_service)],
) -> Model:
    """Import a model from a previously exported zip archive.

    The zip must contain an ``exports/{backend}/metadata.yaml`` file.
    Optionally it may include a ``manifest.json`` with original model
    metadata for reconnecting dataset/snapshot relationships.
    """
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a .zip archive.",
        )

    contents = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(contents))
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid zip archive.",
        )

    with zf:
        # Identify the zip root directory (first path component)
        all_names = zf.namelist()
        if not all_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The zip archive is empty.",
            )

        zip_root = Path(all_names[0]).parts[0]

        # Find metadata.yaml to validate the archive and extract policy
        metadata_entries = [n for n in all_names if n.endswith("metadata.yaml") and "/exports/" in n]
        if not metadata_entries:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid model archive: no exports/*/metadata.yaml found.",
            )

        metadata_path = metadata_entries[0]
        metadata = yaml.safe_load(zf.read(metadata_path))
        policy = _extract_policy_from_metadata(metadata)

        # Read manifest.json if present
        manifest: dict = {}
        manifest_path = f"{zip_root}/manifest.json"
        if manifest_path in all_names:
            manifest = json.loads(zf.read(manifest_path))

        # Create model directory
        settings = get_settings()
        model = Model(
            name=name,
            path="",  # Placeholder, will be set after directory creation
            policy=policy,
            properties=manifest.get("properties", {}),
            project_id=UUID(project_id),
            dataset_id=None,
            snapshot_id=None,
        )
        model_dir = settings.models_dir / str(model.id)
        model_dir.mkdir(parents=True, exist_ok=True)
        model.path = str(model_dir)

        # Try to reconnect dataset/snapshot from manifest
        if manifest.get("original_dataset_id"):
            try:
                dataset = await dataset_service.get_dataset_by_id(UUID(manifest["original_dataset_id"]))
                model.dataset_id = dataset.id
            except Exception:
                pass  # Dataset no longer exists, leave as None

        if manifest.get("original_snapshot_id"):
            try:
                snapshot = await snapshot_service.get_snapshot_by_id(UUID(manifest["original_snapshot_id"]))
                model.snapshot_id = snapshot.id
            except Exception:
                pass  # Snapshot no longer exists, leave as None

        # Extract export artifacts into the model directory
        for entry in all_names:
            entry_path = Path(entry)
            # Skip directories and manifest (already processed)
            if entry.endswith("/"):
                continue

            # Map {zip_root}/exports/... -> {model_dir}/exports/...
            try:
                relative = entry_path.relative_to(zip_root)
            except ValueError:
                continue

            # Only extract files under exports/
            if not str(relative).startswith("exports"):
                continue

            target = model_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(entry))

    # Verify extraction succeeded
    exports_dir = model_dir / "exports"
    if not exports_dir.is_dir() or not any(exports_dir.rglob("*")):
        # Clean up on failure
        import shutil

        shutil.rmtree(model_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to extract model exports from the archive.",
        )

    return await model_service.create_model(model)


@router.delete("")
async def remove_model(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> None:
    """Fetch all projects."""
    model = await model_service.get_model_by_id(model_id)
    if model is None:
        raise ResourceNotFoundError(ResourceType.MODEL, model_id)
    await model_service.delete_model(model)
