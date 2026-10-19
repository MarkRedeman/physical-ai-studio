"""Built-in bimanual StarArm 102-HD leader integration.

The upstream StarArm plugin currently supplies only individual arm drivers.
Studio owns this temporary composition so two HD leaders can teleoperate a
``BimanualWidowXAI`` follower as one 14-joint robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
from physicalai.config import export_config
from physicalai_studio_plugin import RobotAdapterOptions, RobotCatalogDefinition, robot_payload_ui
from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.robot_type import BaseRobot

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import Robot, RobotObservation
    from physicalai_studio_plugin import CatalogRobot, CatalogRobotFactory, PortScanner, SerialPortInfo

BimanualStarArmHDTypes = Literal["Bimanual_StarArm_102_HD_Leader"]
_STARARM_HD_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)


@dataclass
class BimanualStarArmHDObservation:
    """Mapped left/right StarArm observations in WidowX action units."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Alias for joint positions, matching the robot protocol."""
        return self.joint_positions


def _widowx_action(positions: np.ndarray) -> np.ndarray:
    """Map one StarArm HD observation into the matching WidowX action space."""
    action = np.array(positions, dtype=np.float32, copy=True)
    if action.shape != (len(_STARARM_HD_JOINT_NAMES),):
        msg = f"Expected a seven-joint StarArm HD observation, got {action.shape}"
        raise ValueError(msg)
    action[[0, 2, 3, 5]] *= -1
    action[6] = np.clip(action[6] / 55.0 * 0.05, 0.0, 0.05)
    return action


@export_config
class BimanualStarArmHD:
    """Two StarArm 102-HD leaders that emit WidowX-compatible bimanual actions."""

    def __init__(self, left: Robot, right: Robot) -> None:
        if left.joint_names != list(_STARARM_HD_JOINT_NAMES) or right.joint_names != list(_STARARM_HD_JOINT_NAMES):
            msg = "BimanualStarArmHD requires two StarArm 102-HD leader drivers"
            raise ValueError(msg)
        self._left = left
        self._right = right

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Return the physical identities of both leader arms."""
        return tuple(sorted(set(self._left.device_ids) | set(self._right.device_ids)))

    @property
    def joint_names(self) -> list[str]:
        """Return the left/right names used by BimanualWidowXAI."""
        return [f"left_{name}" for name in _STARARM_HD_JOINT_NAMES] + [
            f"right_{name}" for name in _STARARM_HD_JOINT_NAMES
        ]

    def connect(self) -> None:
        """Connect both leaders, cleaning up the left arm if right fails."""
        self._left.connect()
        try:
            self._right.connect()
        except Exception:
            self._left.disconnect()
            raise

    def disconnect(self) -> None:
        """Disconnect both leaders."""
        self._left.disconnect()
        self._right.disconnect()

    def is_connected(self) -> bool:
        """Return whether both leaders are connected."""
        return self._left.is_connected() and self._right.is_connected()

    def get_observation(self) -> RobotObservation:
        """Return both arm poses in the bimanual WidowX action convention."""
        left = self._left.get_observation()
        right = self._right.get_observation()
        return BimanualStarArmHDObservation(
            joint_positions=np.concatenate(
                [_widowx_action(left.joint_positions), _widowx_action(right.joint_positions)]
            ),
            timestamp=left.timestamp,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:  # noqa: ARG002
        """Reject actions: this composite is a leader-only teleoperation input."""
        raise RuntimeError("Cannot send actions to a BimanualStarArmHD leader")


class BimanualStarArmHDPayload(BaseModel):
    """Connection configuration for the left and right StarArm HD leaders."""

    connection_string_left: str = ""
    serial_number_left: str = ""
    connection_string_right: str = ""
    serial_number_right: str = ""
    baudrate: int = Field(default=1_000_000, gt=0)
    unlock_on_connect: bool = True
    reset_multi_turn_on_connect: bool = True
    zero_on_connect: bool = False
    command_interval_ms: int = Field(default=10, ge=0)

    model_config = ConfigDict(
        json_schema_extra=robot_payload_ui(  # pyrefly: ignore[bad-argument-type]
            [
                {
                    "kind": "connection",
                    "label": "Left StarArm HD leader",
                    "device_discovery": True,
                    "bind": {"connection": "connection_string_left", "serial_number": "serial_number_left"},
                },
                {
                    "kind": "connection",
                    "label": "Right StarArm HD leader",
                    "device_discovery": True,
                    "bind": {"connection": "connection_string_right", "serial_number": "serial_number_right"},
                },
            ]
        )
    )

    @model_validator(mode="after")
    def _require_each_connection_identifier(self) -> BimanualStarArmHDPayload:
        if not self.connection_string_left and not self.serial_number_left:
            raise ValueError("A connection_string_left or serial_number_left is required")
        if not self.connection_string_right and not self.serial_number_right:
            raise ValueError("A connection_string_right or serial_number_right is required")
        return self


class BimanualStarArmHDRobot(BaseRobot):
    """Two StarArm 102-HD leaders treated as one bimanual teleop device."""

    type: BimanualStarArmHDTypes
    payload: BimanualStarArmHDPayload


class _StarArmHDDriver(Protocol):
    """The subset of the optional plugin's HD driver used by this composite."""

    def __init__(
        self,
        port: str,
        *,
        baudrate: int,
        unlock_on_connect: bool,
        reset_multi_turn_on_connect: bool,
        zero_on_connect: bool,
        control_mode: Literal["passive"],
        command_interval_ms: int,
    ) -> None: ...

    @property
    def joint_names(self) -> list[str]: ...

    @property
    def device_ids(self) -> tuple[str, ...]: ...

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def get_observation(self) -> RobotObservation: ...
    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None: ...


def _stararm_hd_driver_type() -> type[_StarArmHDDriver]:
    try:
        plugin = import_module("physicalai_stararm_plugin")
    except ImportError as exc:
        msg = "Bimanual StarArm HD requires the physicalai-stararm-plugin package"
        raise RuntimeError(msg) from exc
    return cast("type[_StarArmHDDriver]", plugin.StarArm102HDLeader)


async def _build_bimanual_stararm_hd_driver(
    robot: CatalogRobot[BimanualStarArmHDPayload], factory: CatalogRobotFactory
) -> BimanualStarArmHD:
    """Build both HD leaders after resolving their registered serial devices."""
    payload = robot.payload
    if not isinstance(payload, BimanualStarArmHDPayload):
        raise TypeError(f"Expected BimanualStarArmHDPayload, got {type(payload).__name__}")

    from physicalai_studio_plugin import SerialPortInfo

    left_port, right_port = await _resolve_ports(factory, payload, SerialPortInfo)
    driver_type = _stararm_hd_driver_type()
    left = driver_type(
        port=left_port,
        baudrate=payload.baudrate,
        unlock_on_connect=payload.unlock_on_connect,
        reset_multi_turn_on_connect=payload.reset_multi_turn_on_connect,
        zero_on_connect=payload.zero_on_connect,
        control_mode="passive",
        command_interval_ms=payload.command_interval_ms,
    )
    right = driver_type(
        port=right_port,
        baudrate=payload.baudrate,
        unlock_on_connect=payload.unlock_on_connect,
        reset_multi_turn_on_connect=payload.reset_multi_turn_on_connect,
        zero_on_connect=payload.zero_on_connect,
        control_mode="passive",
        command_interval_ms=payload.command_interval_ms,
    )
    return BimanualStarArmHD(left=left, right=right)


async def _resolve_ports(
    factory: CatalogRobotFactory, payload: BimanualStarArmHDPayload, serial_port_info: type[SerialPortInfo]
) -> tuple[str, str]:
    left = await factory.find_port(
        serial_port_info(
            connection_string=payload.connection_string_left or None,
            serial_number=payload.serial_number_left or None,
        )
    )
    right = await factory.find_port(
        serial_port_info(
            connection_string=payload.connection_string_right or None,
            serial_number=payload.serial_number_right or None,
        )
    )
    if left is None:
        msg = (
            "Could not resolve the left StarArm HD serial port: "
            f"{payload.serial_number_left or payload.connection_string_left}"
        )
        raise ValueError(msg)
    if right is None:
        msg = (
            "Could not resolve the right StarArm HD serial port: "
            f"{payload.serial_number_right or payload.connection_string_right}"
        )
        raise ValueError(msg)
    return left, right


class BimanualStarArmHDProbe:
    """Serial-device probe for the bimanual StarArm HD configuration."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        await manager.find_robots()
        return manager.robots

    async def identify(
        self, payload: BimanualStarArmHDPayload, manager: PortScanner | None = None, joint: str | None = None
    ) -> None:
        """StarArm HD does not implement a safe visual identification action."""
        del payload, manager, joint

    async def is_online(self, payload: BimanualStarArmHDPayload, manager: PortScanner | None = None) -> bool:
        if manager is None:
            return False
        ports = manager.robots
        return _port_present(ports, payload.connection_string_left, payload.serial_number_left) and _port_present(
            ports, payload.connection_string_right, payload.serial_number_right
        )


def _port_present(ports: list[SerialPortInfo], connection_string: str, serial_number: str) -> bool:
    return any(
        (serial_number and port.serial_number == serial_number)
        or (connection_string and port.connection_string == connection_string)
        for port in ports
    )


_BIMANUAL_STARARM_HD_PROBE = BimanualStarArmHDProbe()


def get_definitions() -> list[RobotCatalogDefinition]:
    """Return the temporary built-in bimanual StarArm HD leader definition."""
    return [
        RobotCatalogDefinition(
            type="Bimanual_StarArm_102_HD_Leader",
            display_name="Bimanual Star Arm 102-HD Leader",
            role="leader",
            robot_builder=_build_bimanual_stararm_hd_driver,
            robot_payload=BimanualStarArmHDPayload,
            adapter_options=RobotAdapterOptions(include_velocities=False, external_effort_gain=None),
            probe=_BIMANUAL_STARARM_HD_PROBE,
        )
    ]
