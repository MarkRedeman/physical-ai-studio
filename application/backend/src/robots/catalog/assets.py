from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from schemas.robot import RobotType

if TYPE_CHECKING:
    from .registry import RobotCatalogRegistry


def get_builtin_robot_assets_root() -> Path:
    return Path(__file__).resolve().parents[2] / "static" / "robot-assets"


def resolve_robot_urdf_path(robot_type: RobotType, registry: RobotCatalogRegistry) -> Path:
    relative_path = registry.get_urdf_relative_path(robot_type)
    if relative_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="URDF is unavailable for the requested robot type.")

    return _resolve_robot_path(robot_type=robot_type, relative_path=relative_path, registry=registry)


def resolve_robot_asset_path(robot_type: RobotType, asset_path: str, registry: RobotCatalogRegistry) -> Path:
    relative_path = Path(asset_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")

    package_root = registry.get_package_root(robot_type)
    if package_root is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assets are unavailable for the requested robot type.")

    return _resolve_robot_path(robot_type=robot_type, relative_path=package_root / relative_path, registry=registry)


def _resolve_robot_path(robot_type: RobotType, relative_path: Path, registry: RobotCatalogRegistry) -> Path:
    root = _get_asset_root_for_robot_type(robot_type=robot_type, registry=registry).resolve()

    requested_path = (root / relative_path).resolve()
    if not requested_path.is_relative_to(root):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")
    if not requested_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    return requested_path


def _get_asset_root_for_robot_type(robot_type: RobotType, registry: RobotCatalogRegistry) -> Path:
    if registry.get_asset_source(robot_type) == "builtin":
        return get_builtin_robot_assets_root()

    asset_root_resolver = registry.get_asset_root_resolver(robot_type)
    if asset_root_resolver is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assets are unavailable for the requested robot type.")

    return asset_root_resolver()
