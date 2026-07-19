from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.dependencies import RobotCatalogServiceDep, RobotConnectionManagerDep
from exceptions import ResourceNotFoundError, ResourceType
from robots.catalog.assets import resolve_robot_relative_asset_path, resolve_robot_urdf_path
from robots.catalog.types import RobotCatalogDefinition
from schemas import SerialPortInfo


class RobotCatalogDefinitionResponse(BaseModel):
    type: str = Field(..., description="Stable backend robot type identifier")
    display_name: str = Field(..., description="Human-readable robot type label")
    role: Literal["follower", "leader"] = Field(..., description="Default robot role")
    urdf_path: str = Field(description="URDF URL used by the UI model loader")
    package_map: dict[str, str] = Field(default_factory=dict, description="URDF package name to URL prefix map")
    joint_map: dict[str, list[str]] = Field(
        description="Observation joint name to URDF joint(s) mapping",
    )


def _to_response(definition: RobotCatalogDefinition) -> RobotCatalogDefinitionResponse:
    catalog_root = f"/api/robots/catalog/{definition.type}"

    package_map = {}
    joint_map = {}
    if definition.asset is not None:
        package_map = dict.fromkeys(definition.asset.packages, f"{catalog_root}")
        joint_map = definition.asset.joint_map

    return RobotCatalogDefinitionResponse(
        type=definition.type,
        display_name=definition.display_name,
        role=definition.role,
        urdf_path=f"{catalog_root}/urdf",
        package_map=package_map,
        joint_map=joint_map,
    )


router = APIRouter(prefix="/api/robots/catalog", tags=["Robot Catalog"])


@router.get("")
async def list_robot_catalog(catalog_service: RobotCatalogServiceDep) -> list[RobotCatalogDefinitionResponse]:
    """List robot catalog definitions exposed to the UI."""
    return [_to_response(definition) for definition in catalog_service.list_entries()]


@router.get("/{robot_type}/discover")
async def discover_robots(
    catalog_service: RobotCatalogServiceDep,
    robot_manager: RobotConnectionManagerDep,
    robot_type: str,
) -> list[SerialPortInfo]:
    """Discover connected devices for a robot type."""
    from loguru import logger

    definition = catalog_service.get_definition(robot_type)

    logger.info("Discover from definition {} as {}", robot_type, definition)
    if definition.probe is None:
        return []
    return await definition.probe.discover(robot_manager)


@router.post("/{robot_type}/identify")
async def identify_robot(
    catalog_service: RobotCatalogServiceDep,
    robot_manager: RobotConnectionManagerDep,
    robot_type: str,
    payload: dict[str, Any],
    joint: str | None = None,
) -> None:
    """Visually identify a robot by moving a joint or gripper."""
    definition = catalog_service.get_definition(robot_type)
    if definition.probe is None:
        raise ResourceNotFoundError(
            resource_type=ResourceType.ROBOT,
            resource_id=robot_type,
            message=f"Robot type {robot_type} does not support identification.",
        )
    await definition.probe.identify(payload, robot_manager, joint)


@router.post("/{robot_type}/is-online")
async def check_robot_online(
    catalog_service: RobotCatalogServiceDep,
    robot_type: str,
    payload: dict[str, Any],
) -> bool:
    """Check if a robot is currently online/reachable."""
    definition = catalog_service.get_definition(robot_type)
    if definition.probe is None:
        return False
    return await definition.probe.is_online(payload)


@router.get("/{robot_type}/urdf")
async def get_robot_catalog_urdf(catalog_service: RobotCatalogServiceDep, robot_type: str) -> FileResponse:
    """Return the URDF file for a catalog robot type."""
    definition = catalog_service.get_definition(robot_type)

    resolved_path = resolve_robot_urdf_path(definition)
    return FileResponse(resolved_path)


@router.get("/{robot_type}/schema")
async def get_robot_catalog_schema(
    catalog_service: RobotCatalogServiceDep,
    robot_type: str,
) -> dict[str, Any]:
    """Return the Pydantic JSON Schema for a catalog robot payload."""
    definition = catalog_service.get_definition(robot_type)
    payload = definition.robot_payload
    if payload is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Robot type {robot_type} has no payload schema")
    return payload.model_json_schema()


@router.get("/{robot_type}/{asset_path:path}")
async def get_robot_catalog_asset(
    catalog_service: RobotCatalogServiceDep,
    robot_type: str,
    asset_path: Path,
) -> FileResponse:
    """Return an asset referenced by a relative URDF URI."""
    definition = catalog_service.get_definition(robot_type)

    resolved_path = resolve_robot_relative_asset_path(definition, asset_path=asset_path)
    return FileResponse(resolved_path)
