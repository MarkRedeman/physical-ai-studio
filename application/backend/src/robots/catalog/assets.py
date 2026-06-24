from pathlib import Path

from schemas.robot import RobotType

BASE_ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "static" / "robot-assets"


def _get_asset_root_for_robot_type(registry, robot_type: RobotType) -> Path | None:
    definition = registry.get_definition(robot_type)
    if definition is None:
        return None

    resolver = definition.asset_root_resolver
    if resolver is not None:
        return BASE_ASSET_DIR / resolver()

    source = definition.asset_source
    if source == "builtin":
        package_root = definition.package_root
        if package_root:
            return BASE_ASSET_DIR / package_root
    elif source == "plugin":
        urdf_relative = definition.urdf_relative_path
        if urdf_relative:
            return BASE_ASSET_DIR / urdf_relative.parent

    return None


def resolve_robot_urdf_path(registry, robot_type: RobotType) -> Path | None:
    definition = registry.get_definition(robot_type)
    if definition is None or definition.urdf_relative_path is None:
        return None

    asset_root = _get_asset_root_for_robot_type(registry, robot_type)
    if asset_root is None:
        return None

    return asset_root / definition.urdf_relative_path


def resolve_robot_asset_path(registry, robot_type: RobotType, asset_path: str) -> Path | None:
    asset_root = _get_asset_root_for_robot_type(registry, robot_type)
    if asset_root is None:
        return None

    resolved = (asset_root / asset_path).resolve()

    if not str(resolved).startswith(str(asset_root.resolve())):
        return None

    if not resolved.exists():
        return None

    return resolved
