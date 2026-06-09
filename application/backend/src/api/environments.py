from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from api.dependencies import (
    get_environment_id,
    get_environment_service,
    get_project_id,
    get_robot_client_factory,
)
from schemas.environment import (
    Environment,
    EnvironmentManifestCameraEntry,
    EnvironmentManifestResponse,
    EnvironmentManifestRobotEntry,
    EnvironmentWithRelations,
)
from utils.camera_factory import is_migrated
from services.environment_service import EnvironmentService
from robots.robot_client_factory import RobotClientFactory

router = APIRouter(prefix="/api/projects/{project_id}/environments", tags=["Project Environments"])

ProjectID = Annotated[UUID, Depends(get_project_id)]


@router.get("")
async def list_project_environments(
    project_id: ProjectID,
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> list[Environment]:
    """Fetch all environments."""
    return await environment_service.get_environment_list(project_id)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project_environment(
    project_id: ProjectID,
    environment: Environment,
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> Environment:
    """Create a new environment."""
    return await environment_service.create_environment(project_id, environment)


@router.get("/{environment_id}")
async def get_project_environment(
    project_id: Annotated[UUID, Depends(get_project_id)],
    environment_id: Annotated[UUID, Depends(get_environment_id)],
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> EnvironmentWithRelations:
    """Get environment by id with eager loaded robots and cameras."""
    return await environment_service.get_environment_by_id(project_id, environment_id)


@router.get("/{environment_id}/manifest")
async def get_environment_manifest(
    project_id: Annotated[UUID, Depends(get_project_id)],
    environment_id: Annotated[UUID, Depends(get_environment_id)],
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
    robot_client_factory: Annotated[RobotClientFactory, Depends(get_robot_client_factory)],
) -> EnvironmentManifestResponse:
    """Get environment manifest for compatibility checks."""
    environment = await environment_service.get_environment_by_id(project_id, environment_id)

    robots: list[EnvironmentManifestRobotEntry] = []
    for robot_with_teleop in environment.robots:
        robot = robot_with_teleop.robot
        client = await robot_client_factory.build(robot)
        robots.append(
            EnvironmentManifestRobotEntry(
                name=robot.name,
                robot_type=str(robot.type),
                features=client.features(),
                fps=30,
            )
        )

    cameras: list[EnvironmentManifestCameraEntry] = []
    for camera in environment.cameras:
        if not is_migrated(camera.driver):
            continue

        resolution: tuple[int, int] | None = None
        fps: int | None = None
        payload = camera.payload
        if payload is not None:
            width = getattr(payload, "width", None)
            height = getattr(payload, "height", None)
            fps_value = getattr(payload, "fps", None)
            if isinstance(width, int) and isinstance(height, int):
                resolution = (width, height)
            if isinstance(fps_value, int):
                fps = fps_value

        cameras.append(
            EnvironmentManifestCameraEntry(
                name=camera.name,
                driver=camera.driver,
                resolution=resolution,
                fps=fps,
            )
        )

    return EnvironmentManifestResponse(robots=robots, cameras=cameras)


@router.put("/{environment_id}")
async def update_project_environment(
    project_id: Annotated[UUID, Depends(get_project_id)],
    environment_id: Annotated[UUID, Depends(get_environment_id)],
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
    environment: Environment,
) -> EnvironmentWithRelations:
    """Update environment."""
    environment_with_id = environment.model_copy(update={"id": environment_id})

    return await environment_service.update_environment(
        project_id,
        environment_with_id,
    )


@router.delete("/{environment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_environment(
    project_id: Annotated[UUID, Depends(get_project_id)],
    environment_id: Annotated[UUID, Depends(get_environment_id)],
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> None:
    """Delete an environment."""
    await environment_service.delete_environment(project_id, environment_id)
