"""Plugin management API endpoints."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.dependencies import AsyncSessionDep, HealthServiceDep, PluginManagerDep
from exceptions import ResourceInUseError, ResourceType
from plugins.plugin_manager import (
    PluginInfo,
    PluginRestoreResult,
    PluginRestoreStatus,
    PluginRobot,
    find_robot_types_in_use_async,
)

router = APIRouter(prefix="/api/plugins", tags=["Plugins"])


class PluginRobotResponse(BaseModel):
    type: str = Field(..., description="Stable robot type identifier")
    display_name: str = Field(..., description="Human-readable robot type label")
    role: Literal["follower", "leader"] = Field(..., description="Default robot role")
    installed: bool = Field(..., description="Whether the robot type is currently available in the catalog")


class PluginResponse(BaseModel):
    id: str = Field(..., description="Python distribution name")
    name: str = Field(..., description="Display name shown in the UI")
    description: str = Field(..., description="Short plugin description")
    repo_url: str | None = Field(default=None, description="Project repository URL")
    installed: bool = Field(..., description="Whether the plugin distribution is installed")
    installed_version: str | None = Field(default=None, description="Installed plugin version")
    in_use_robot_count: int = Field(..., description="Number of persisted robots using this plugin's robot types")
    robots: list[PluginRobotResponse] = Field(..., description="Robot types contributed by the plugin")


class PluginOperationResponse(BaseModel):
    restart_required: bool = Field(default=True, description="A server restart is required to activate the change")


class PluginRestoreStatusResponse(BaseModel):
    needs_restore: bool = Field(..., description="Whether known recorded plugins are missing")
    missing_plugin_ids: list[str] = Field(..., description="Known recorded plugins missing from the environment")
    unknown_plugin_ids: list[str] = Field(..., description="Recorded plugin IDs no longer present in the manifest")


class PluginRestoreResponse(BaseModel):
    restored_plugin_ids: list[str] = Field(..., description="Plugins restored during this operation")
    failed_plugin_ids: list[str] = Field(..., description="Plugins that could not be restored")
    unknown_plugin_ids: list[str] = Field(..., description="Recorded plugin IDs no longer present in the manifest")
    restart_required: bool = Field(..., description="Whether a server restart is required to activate the changes")


def _to_robot_response(robot: PluginRobot) -> PluginRobotResponse:
    return PluginRobotResponse(
        type=robot.type,
        display_name=robot.display_name,
        role=robot.role,
        installed=robot.installed,
    )


def _to_response(plugin: PluginInfo, in_use_robot_count: int) -> PluginResponse:
    return PluginResponse(
        id=plugin.id,
        name=plugin.name,
        description=plugin.description,
        repo_url=plugin.repo_url,
        installed=plugin.installed,
        installed_version=plugin.installed_version,
        in_use_robot_count=in_use_robot_count,
        robots=[_to_robot_response(robot) for robot in plugin.robots],
    )


def _to_restore_status_response(status: PluginRestoreStatus) -> PluginRestoreStatusResponse:
    return PluginRestoreStatusResponse(
        needs_restore=status.needs_restore,
        missing_plugin_ids=status.missing_plugin_ids,
        unknown_plugin_ids=status.unknown_plugin_ids,
    )


def _to_restore_response(result: PluginRestoreResult) -> PluginRestoreResponse:
    return PluginRestoreResponse(
        restored_plugin_ids=result.restored_plugin_ids,
        failed_plugin_ids=result.failed_plugin_ids,
        unknown_plugin_ids=result.unknown_plugin_ids,
        restart_required=result.restart_required,
    )


@router.get("")
async def list_plugins(
    plugin_manager: PluginManagerDep,
    session: AsyncSessionDep,
) -> list[PluginResponse]:
    """List available and installed plugins with their robot types."""
    plugins = plugin_manager.list_plugins()
    robot_types_by_plugin = {plugin.id: [robot.type for robot in plugin.robots] for plugin in plugins}
    all_robot_types = sorted({type_ for types in robot_types_by_plugin.values() for type_ in types})
    in_use_robot_types = set(await find_robot_types_in_use_async(session, all_robot_types))

    return [
        _to_response(
            plugin,
            in_use_robot_count=sum(1 for type_ in robot_types_by_plugin[plugin.id] if type_ in in_use_robot_types),
        )
        for plugin in plugins
    ]


@router.get("/restore-status")
async def get_restore_status(plugin_manager: PluginManagerDep) -> PluginRestoreStatusResponse:
    """Report whether recorded plugins are missing from the active environment."""
    return _to_restore_status_response(plugin_manager.get_restore_status())


@router.post(":restore")
async def restore_plugins(
    plugin_manager: PluginManagerDep,
    health_service: HealthServiceDep,
) -> PluginRestoreResponse:
    """Restore recorded plugins and report whether a restart is required."""
    result = await plugin_manager.restore_installed()
    if result.restart_required:
        health_service.mark_plugin_restart_required()
    return _to_restore_response(result)


@router.post("/{plugin_id}")
async def install_plugin(
    plugin_id: str,
    plugin_manager: PluginManagerDep,
    health_service: HealthServiceDep,
) -> PluginOperationResponse:
    """Install a plugin distribution and require a server restart to activate."""
    await plugin_manager.install(plugin_id)
    health_service.mark_plugin_restart_required()
    return PluginOperationResponse()


@router.delete("/{plugin_id}")
async def uninstall_plugin(
    plugin_id: str,
    plugin_manager: PluginManagerDep,
    session: AsyncSessionDep,
    health_service: HealthServiceDep,
) -> PluginOperationResponse:
    """Uninstall a plugin distribution after checking no robots reference its types."""
    in_use_robot_types = await find_robot_types_in_use_async(session, plugin_manager.robot_types(plugin_id))
    if in_use_robot_types:
        plugin = plugin_manager.get(plugin_id)
        raise ResourceInUseError(
            ResourceType.PLUGIN,
            plugin_id,
            message=(
                f"Cannot uninstall '{plugin.name}': {len(in_use_robot_types)} robot(s) use type(s) "
                f"{', '.join(in_use_robot_types)}. Delete those robots first."
            ),
        )
    await plugin_manager.uninstall(plugin_id)
    health_service.mark_plugin_restart_required()
    return PluginOperationResponse()
