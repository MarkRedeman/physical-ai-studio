from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from robots.catalog import resolve_robot_asset_path, resolve_robot_urdf_path
from schemas.robot import RobotType
from schemas.robot_catalog import RobotCatalogDiscoverResponse, RobotCatalogEntry, RobotCatalogOnlineResponse
from services.robot_catalog_service import RobotCatalogServiceDep

router = APIRouter(prefix="/api/robots/catalog", tags=["Robot Catalog"])


@router.get("")
async def list_catalog(service: RobotCatalogServiceDep) -> list[RobotCatalogEntry]:
    return service.list_entries()


@router.get("/{robot_type}")
async def get_catalog_entry(robot_type: RobotType, service: RobotCatalogServiceDep) -> RobotCatalogEntry:
    entry = service.get_entry(robot_type)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Robot type not found: {robot_type}")
    return entry


@router.get("/{robot_type}/online")
async def is_robot_type_online(robot_type: RobotType, service: RobotCatalogServiceDep) -> RobotCatalogOnlineResponse:
    online = await service.is_type_online(robot_type)
    return RobotCatalogOnlineResponse(online=online)


@router.get("/{robot_type}/discover")
async def discover_robots(robot_type: RobotType, service: RobotCatalogServiceDep) -> RobotCatalogDiscoverResponse:
    ports = await service.discover_devices(robot_type)
    return RobotCatalogDiscoverResponse(ports=ports)


@router.get("/{robot_type}/urdf")
async def get_robot_urdf(robot_type: RobotType) -> FileResponse:
    from robots.catalog.registry import RobotCatalogRegistry

    registry = RobotCatalogRegistry()
    path = resolve_robot_urdf_path(registry, robot_type)
    if path is None or not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="URDF not found")
    return FileResponse(str(path), media_type="application/xml")


@router.get("/{robot_type}/{asset_path:path}")
async def get_robot_asset(robot_type: RobotType, asset_path: str) -> FileResponse:
    from robots.catalog.registry import RobotCatalogRegistry

    registry = RobotCatalogRegistry()
    path = resolve_robot_asset_path(registry, robot_type, asset_path)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    return FileResponse(str(path))
