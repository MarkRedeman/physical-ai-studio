from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from api.dependencies import get_robot_catalog_service
from robots.catalog import resolve_robot_asset_path, resolve_robot_urdf_path
from schemas.robot import RobotType
from schemas.robot_catalog import RobotCatalogDiscoverResponse, RobotCatalogEntry, RobotCatalogOnlineResponse
from services import RobotCatalogService

router = APIRouter(prefix="/api/robots/catalog", tags=["Robot Catalog"])

CatalogServiceDep = Annotated[RobotCatalogService, Depends(get_robot_catalog_service)]


@router.get("")
async def list_robot_catalog(catalog_service: CatalogServiceDep) -> list[RobotCatalogEntry]:
    return catalog_service.list_entries()


@router.get("/{robot_type}")
async def get_robot_catalog_entry(robot_type: RobotType, catalog_service: CatalogServiceDep) -> RobotCatalogEntry:
    entry = catalog_service.get_entry(robot_type)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Robot type is not part of the catalog.")
    return entry


@router.get("/{robot_type}/online")
async def get_robot_catalog_online_status(
    robot_type: RobotType, catalog_service: CatalogServiceDep
) -> RobotCatalogOnlineResponse:
    entry = catalog_service.get_entry(robot_type)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Robot type is not part of the catalog.")

    return RobotCatalogOnlineResponse(type=robot_type, online=await catalog_service.is_type_online(robot_type))


@router.get("/{robot_type}/discover")
async def discover_robot_catalog_devices(
    robot_type: RobotType, catalog_service: CatalogServiceDep
) -> RobotCatalogDiscoverResponse:
    entry = catalog_service.get_entry(robot_type)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Robot type is not part of the catalog.")

    devices = await catalog_service.discover_devices(robot_type)
    return RobotCatalogDiscoverResponse(type=robot_type, devices=devices)


@router.get("/{robot_type}/urdf")
async def get_robot_catalog_urdf(robot_type: RobotType, catalog_service: CatalogServiceDep) -> FileResponse:
    entry = catalog_service.get_entry(robot_type)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Robot type is not part of the catalog.")

    resolved_path = resolve_robot_urdf_path(robot_type=robot_type, registry=catalog_service.registry)
    return FileResponse(resolved_path)


@router.get("/{robot_type}/{asset_path:path}")
async def get_robot_catalog_asset(
    robot_type: RobotType,
    asset_path: str,
    catalog_service: CatalogServiceDep,
) -> FileResponse:
    entry = catalog_service.get_entry(robot_type)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Robot type is not part of the catalog.")

    resolved_path = resolve_robot_asset_path(robot_type=robot_type, asset_path=asset_path, registry=catalog_service.registry)
    return FileResponse(resolved_path)
