from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response, status

from api.dependencies import get_episode_thumbnail_service, get_model_service, get_project_id, get_project_service
from internal_datasets.utils import get_internal_read_dataset
from schemas import Model, Project
from services import EpisodeThumbnailService, ModelService, ProjectService

router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.get("")
async def list_projects(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> list[Project]:
    """Fetch all projects."""
    return await project_service.get_project_list()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    project: Project,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Create a new project."""
    return await project_service.create_project(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> None:
    """Delete a project."""
    await project_service.delete_project(project_id)


@router.get("/{project_id}")
async def get_project(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Get project by id."""
    return await project_service.get_project_by_id(project_id)


@router.get("/{project_id}/models")
async def get_project_models(
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> list[Model]:
    """Get all models of a project."""
    return await model_service.get_project_models(project_id)


@router.get("/{project_id}/tasks")
async def get_tasks_for_dataset(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> dict[str, list[str]]:
    """Get all dataset tasks of a project."""
    project = await project_service.get_project_by_id(project_id)
    res = {}

    for dataset in project.datasets:
        res[dataset.name] = get_internal_read_dataset(dataset).get_tasks()

    return res


@router.get("/{project_id}/thumbnail")
async def get_project_thumbnail(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    thumbnail_service: Annotated[EpisodeThumbnailService, Depends(get_episode_thumbnail_service)],
    request: Request,
    width: Annotated[int, Query(ge=32, le=1920)] = 156,
    height: Annotated[int, Query(ge=32, le=1080)] = 156,
) -> Response:
    """Get a project thumbnail from the first frame of the first episode in the first dataset."""
    project = await project_service.get_project_by_id(project_id)

    if len(project.datasets) == 0:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    dataset = project.datasets[0]
    internal_dataset = get_internal_read_dataset(dataset)
    episode_infos = internal_dataset.get_episode_infos()

    if len(episode_infos) == 0:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    episode_index = episode_infos[0].episode_index
    thumbnail = thumbnail_service.get_thumbnail(
        dataset_id=dataset.id,
        dataset=internal_dataset,
        episode_index=episode_index,
        width=width,
        height=height,
    )

    if thumbnail is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    cache_headers = {
        "Cache-Control": "public, max-age=3600, stale-while-revalidate=86400",
        "ETag": thumbnail.etag,
        "Last-Modified": thumbnail.last_modified,
        "Vary": "Accept",
    }

    request_etag = request.headers.get("if-none-match")
    request_last_modified = request.headers.get("if-modified-since")
    if request_etag == thumbnail.etag or request_last_modified == thumbnail.last_modified:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=cache_headers)

    return Response(content=thumbnail.content, media_type="image/png", headers=cache_headers)
