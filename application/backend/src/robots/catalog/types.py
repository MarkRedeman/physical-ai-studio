from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, runtime_checkable

from physicalai.robot.interface import Robot as PhysicalAIRobot

from schemas import SerialPortInfo

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from pydantic import BaseModel

    from schemas.calibration import Calibration


@dataclass(frozen=True)
class RobotAdapterOptions:
    include_velocities: bool = False
    goal_time_scale: float = 1.0
    external_effort_gain: float | None = 0.1


@dataclass(frozen=True)
class RobotAsset:
    """Filesystem configuration for a robot's visual model and ROS packages."""

    urdf_relative_path: Path
    packages: dict[str, Path]
    joint_map: dict[str, list[str]]
    root_resolver: Callable[[], Path] | None = None


@runtime_checkable
class RobotProbe(Protocol):
    """Hardware interaction interface for a robot type.

    Each built-in or plugin-provided robot type implements this protocol
    to encapsulate device discovery, visual identification, and online
    status checking. External packages implement the protocol by defining
    a class with these three methods.
    """

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]: ...

    async def identify(
        self,
        payload: dict[str, Any],
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None: ...

    async def is_online(self, payload: dict[str, Any], manager: PortScanner | None = None) -> bool: ...


class PortScanner(Protocol):
    """Duck-type for serial port scanners (e.g. RobotConnectionManager).

    Used by RobotProbe to accept scan results without importing
    RobotConnectionManager directly (which would create circular deps).
    """

    async def find_robots(self) -> None: ...

    @property
    def robots(self) -> list[SerialPortInfo]: ...


class CatalogRobotFactory(Protocol):
    async def find_so101_port(self, robot: Any) -> str: ...

    async def find_port_by_serial(self, serial_number: str) -> str | None: ...

    async def get_calibration_by_id(self, calibration_id: UUID | None) -> Calibration | None: ...


_PayloadT = TypeVar("_PayloadT")


class PayloadContainer(Protocol[_PayloadT]):
    payload: _PayloadT


class CatalogRobot(PayloadContainer[_PayloadT], Protocol[_PayloadT]):
    type: str
    active_calibration_id: UUID | None


_RobotT = TypeVar("_RobotT", bound=CatalogRobot[Any])
_FactoryT = TypeVar("_FactoryT", bound="CatalogRobotFactory")


BuildRobotCallable = Callable[[_RobotT, _FactoryT], Awaitable[PhysicalAIRobot]]


@dataclass
class RobotCatalogDefinition:
    type: str
    display_name: str
    role: Literal["follower", "leader"]
    robot_builder: BuildRobotCallable
    robot_payload: type[BaseModel]
    asset: RobotAsset | None

    adapter_options: RobotAdapterOptions = field(default_factory=RobotAdapterOptions)
    probe: RobotProbe | None = None

    @property
    def robot_type(self) -> str:
        return self.type
