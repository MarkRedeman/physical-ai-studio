import io
import zipfile
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_dataset_service, get_model_service, validate_uuid
from exceptions import ResourceNotFoundError, ResourceType
from internal_datasets.utils import get_internal_dataset
from schemas import Model
from services import DatasetService, ModelService

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("/{model_id}")
async def get_model_by_id(
    model_id: Annotated[UUID, Depends(validate_uuid)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> Model:
    """Get model by id."""
    return await model_service.get_model_by_id(model_id)


@router.get("/{model_id}/tasks")
async def get_tasks_of_model(
    model_id: Annotated[UUID, Depends(validate_uuid)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[str]:
    """Get availabe tasks for model."""
    model = await model_service.get_model_by_id(model_id)
    dataset = await dataset_service.get_dataset_by_id(model.dataset_id)
    return get_internal_dataset(dataset).get_tasks()


@router.get("/{model_id}:export")
async def export_model(
    model_id: Annotated[UUID, Depends(validate_uuid)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> StreamingResponse:
    """Export a model as a zip archive containing the model's export artifacts.

    The zip contains the exports/ subtree of the model directory, structured as
    ``{model_name}/exports/{backend}/{files}``. This layout is designed so that
    a future import endpoint can extract the archive directly into a new model
    directory.
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

    def _generate_zip():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
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


@router.delete("")
async def remove_model(
    model_id: Annotated[UUID, Depends(validate_uuid)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> None:
    """Fetch all projects."""
    model = await model_service.get_model_by_id(model_id)
    if model is None:
        raise ResourceNotFoundError(ResourceType.MODEL, model_id)
    await model_service.delete_model(model)
