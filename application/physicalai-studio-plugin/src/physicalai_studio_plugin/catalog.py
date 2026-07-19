from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, Type, TypeVar

from physicalai.robot.interface import Robot as PhysicalAIRobot

from .assets import RobotAsset
from .factory import CatalogRobotFactory
from .probe import RobotProbe

if TYPE_CHECKING:
    from uuid import UUID

    from pydantic import BaseModel


@dataclass(frozen=True)
class RobotAdapterOptions:
    include_velocities: bool = False
    goal_time_scale: float = 1.0
    external_effort_gain: float | None = 0.1


_PayloadT = TypeVar("_PayloadT")


class PayloadContainer(Protocol[_PayloadT]):
    payload: _PayloadT


class CatalogRobot(PayloadContainer[_PayloadT], Protocol[_PayloadT]):
    type: str
    active_calibration_id: UUID | None


_RobotT = TypeVar("_RobotT", bound=CatalogRobot[Any])
_FactoryT = TypeVar("_FactoryT", bound=CatalogRobotFactory)


BuildRobotCallable = Callable[[_RobotT, _FactoryT], Awaitable[PhysicalAIRobot]]


@dataclass
class RobotCatalogDefinition:
    type: str
    display_name: str
    role: Literal["follower", "leader"]
    robot_builder: BuildRobotCallable | None = None
    robot_payload: Type[BaseModel] | None = None  # noqa: UP006
    asset: RobotAsset | None = None

    adapter_options: RobotAdapterOptions = field(default_factory=RobotAdapterOptions)
    probe: RobotProbe | None = None

    @property
    def robot_type(self) -> str:
        return self.type
