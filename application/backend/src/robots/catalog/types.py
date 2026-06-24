from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from physicalai.robot.interface import Robot as PhysicalAIRobot
from pydantic import BaseModel

from schemas.robot import Robot as SchemaRobot
from schemas.robot import RobotType, SerialPortInfo
from schemas.robot_catalog import RobotCatalogEntry

if TYPE_CHECKING:
    from robots.robot_client_factory import RobotClientFactory

    from .registry import RobotCatalogRegistry

AssetSource = Literal["builtin", "plugin"]
DiscoverDevicesCallable = Callable[[list[SerialPortInfo]], Awaitable[list[SerialPortInfo]]]
AssetRootResolver = Callable[[], Path]
BuildRobotCallable = Callable[[SchemaRobot, RobotClientFactory], Awaitable[PhysicalAIRobot]]
PayloadModelType = type[BaseModel]


@dataclass(frozen=True)
class RobotAdapterOptions:
    include_velocities: bool = False
    goal_time_scale: float = 1.0
    external_effort_gain: float | None = 0.1


class RegisterPluginCallable(Protocol):
    def __call__(self, registry: RobotCatalogRegistry) -> None: ...


class CatalogPluginModule(Protocol):
    def register_physicalai_studio_plugin(self, registry: RobotCatalogRegistry) -> None: ...


@dataclass(frozen=True)
class RobotCatalogDefinition:
    entry: RobotCatalogEntry
    urdf_relative_path: Path | None
    package_root: Path | None
    asset_source: AssetSource
    asset_root_resolver: AssetRootResolver | None
    discover_devices: DiscoverDevicesCallable
    robot_builder: BuildRobotCallable | None = None
    payload_model: PayloadModelType | None = None
    adapter_options: RobotAdapterOptions = RobotAdapterOptions()

    @property
    def robot_type(self) -> RobotType:
        return self.entry.type
