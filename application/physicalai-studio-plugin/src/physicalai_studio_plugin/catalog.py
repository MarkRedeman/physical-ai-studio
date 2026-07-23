from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar

from physicalai.robot.interface import Robot as PhysicalAIRobot
from pydantic import BaseModel

from .assets import RobotAsset
from .factory import CatalogRobotFactory
from .probe import RobotProbe


@dataclass(frozen=True)
class RobotAdapterOptions:
    include_velocities: bool = False
    goal_time_scale: float = 1.0
    external_effort_gain: float | None = 0.1


_PayloadT = TypeVar("_PayloadT", bound=BaseModel)


class PayloadContainer(Protocol[_PayloadT]):
    payload: _PayloadT


class CatalogRobot(PayloadContainer[_PayloadT], Protocol[_PayloadT]):
    type: str


_RobotT = TypeVar("_RobotT", bound=CatalogRobot[Any])
_FactoryT = TypeVar("_FactoryT", bound=CatalogRobotFactory)


BuildRobotCallable = Callable[[_RobotT, _FactoryT], Awaitable[PhysicalAIRobot]]


@dataclass
class RobotCatalogDefinition(Generic[_PayloadT]):
    type: str
    display_name: str
    role: Literal["follower", "leader"]
    robot_builder: BuildRobotCallable | None = None
    robot_payload: type[_PayloadT] | None = None
    asset: RobotAsset | None = None

    adapter_options: RobotAdapterOptions = field(default_factory=RobotAdapterOptions)
    probe: RobotProbe[_PayloadT] | None = None

    @property
    def robot_type(self) -> str:
        return self.type
