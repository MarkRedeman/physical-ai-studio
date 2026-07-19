from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from physicalai.robot.so101 import SO101, SO101Calibration
from pydantic import BaseModel, ConfigDict, Field, model_validator

from exceptions import ResourceNotFoundError, ResourceType
from schemas import SerialPortInfo
from schemas.robot_type import BaseRobot

from .types import RobotAdapterOptions, RobotAsset, RobotCatalogDefinition

SO101Types = Literal["SO101_Follower", "SO101_Leader"]

if TYPE_CHECKING:
    from schemas.calibration import Calibration
    from schemas.robot import Robot

    from .types import CatalogRobot, CatalogRobotFactory, PortScanner


class SO101RobotPayload(BaseModel):
    """Connection configuration for SO-101 serial robots."""

    connection_string: str = Field(
        default="",
        description="Serial port path; leave empty to auto-discover via serial_number",
    )
    serial_number: str = Field(default="", description="USB serial number of the robot (when available)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "connection_string": "",
                "serial_number": "SO101-2024-001",
            },
        },
    )

    @model_validator(mode="after")
    def validate_identifier(self) -> SO101RobotPayload:
        if self.connection_string == "" and self.serial_number == "":
            raise ValueError("Either serial_number or connection_string is required for SO101 robots")
        return self


class SO101Robot(BaseRobot):
    """SO-101 follower or leader robot using a serial connection."""

    type: SO101Types = Field(..., description="Type of robot configuration")
    payload: SO101RobotPayload = Field(..., description="SO-101 connection configuration")


_SO101_TO_URDF = {
    "shoulder_pan.pos": ["shoulder_pan"],
    "shoulder_lift.pos": ["shoulder_lift"],
    "elbow_flex.pos": ["elbow_flex"],
    "wrist_flex.pos": ["wrist_flex"],
    "wrist_roll.pos": ["wrist_roll"],
    "gripper.pos": ["gripper"],
}

_SO101_ASSET = RobotAsset(
    urdf_relative_path=Path("SO101/so101_new_calib.urdf"),
    packages={"SO101": Path("SO101")},
    joint_map=_SO101_TO_URDF,
)


def _to_so101_calibration(calibration: Calibration) -> SO101Calibration:
    return SO101Calibration.from_dict(
        {
            name: {
                "id": val.id,
                "drive_mode": val.drive_mode,
                "homing_offset": val.homing_offset,
                "range_min": val.range_min,
                "range_max": val.range_max,
            }
            for name, val in calibration.values.items()
        }
    )


async def _build_so101_driver(robot: CatalogRobot[SO101RobotPayload], factory: CatalogRobotFactory) -> SO101:
    if not isinstance(robot.payload, SO101RobotPayload):
        raise TypeError("Expected SO101Robot")
    port = await factory.find_so101_port(robot)
    calibration = await factory.get_calibration_by_id(robot.active_calibration_id)

    if calibration is None:
        raise ResourceNotFoundError(ResourceType.ROBOT_CALIBRATION, robot.payload.serial_number)

    role = "follower" if robot.type == "SO101_Follower" else "leader"
    return SO101(port=port, calibration=_to_so101_calibration(calibration), role=role, unit="normalized")


def serial_port_from_so101(robot: SO101Robot) -> SerialPortInfo:
    """Build a serial identity from an SO101 robot configuration."""
    connection_string = robot.payload.connection_string or None
    serial_number = robot.payload.serial_number or None
    return SerialPortInfo(connection_string=connection_string, serial_number=serial_number)


def _resolve_serial_port(discovered: list[SerialPortInfo], target: SerialPortInfo) -> str | None:
    if target.serial_number is not None:
        for serial_port in discovered:
            if serial_port.serial_number == target.serial_number:
                return serial_port.connection_string
        return None

    for serial_port in discovered:
        if serial_port.connection_string == target.connection_string:
            return serial_port.connection_string
    return None


async def find_so101_port(
    manager: PortScanner,
    serial_port: SerialPortInfo,
) -> str | None:
    """Find the current port for an SO101 robot by serial number or configured port."""
    port = _resolve_serial_port(manager.robots, serial_port)
    if port is not None:
        return port

    await manager.find_robots()
    return _resolve_serial_port(manager.robots, serial_port)


async def identify_so101_robot_visually(
    manager: PortScanner,
    robot: Robot,
    joint: str | None = None,
) -> None:
    """Identify the robot by moving the joint from current to min to max to initial position."""
    import asyncio

    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    if not isinstance(robot.payload, SO101RobotPayload):
        raise ValueError(f"Trying to identify unsupported robot: {robot.type}")

    if joint is None:
        joint = "gripper"

    connection_string = await find_so101_port(manager, serial_port_from_so101(robot))

    if connection_string is None:
        if robot.payload.serial_number:
            raise ValueError(f"Could not find the serial port for serial number {robot.payload.serial_number}")
        raise ValueError("Could not resolve a serial port from connection_string")
    connection = SOFollower(SOFollowerRobotConfig(port=connection_string))
    connection.bus.connect()

    PRESENT_POSITION_KEY = "Present_Position"
    GOAL_POSITION_KEY = "Goal_Position"

    current_position = connection.bus.sync_read(PRESENT_POSITION_KEY, normalize=False)
    gripper_calibration = connection.bus.read_calibration()[joint]
    connection.bus.write(GOAL_POSITION_KEY, joint, gripper_calibration.range_min, normalize=False)
    await asyncio.sleep(1)
    connection.bus.write(GOAL_POSITION_KEY, joint, gripper_calibration.range_max, normalize=False)
    await asyncio.sleep(1)
    connection.bus.write(GOAL_POSITION_KEY, joint, current_position[joint], normalize=False)
    await asyncio.sleep(1)
    connection.bus.disconnect()


class SO101Probe:
    """Probe for SO101 robots — serial port discovery + joint identification."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        await manager.find_robots()
        return manager.robots

    async def identify(
        self,
        payload: dict[str, Any],
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None:
        if manager is None:
            raise ValueError("PortScanner required for SO101 identification")

        robot_payload = SO101RobotPayload(**payload)
        now = datetime.now()
        robot = SO101Robot(
            id=UUID(int=0),
            name="",
            type="SO101_Follower",
            payload=robot_payload,
            active_calibration_id=None,
            created_at=now,
            updated_at=now,
        )
        await identify_so101_robot_visually(manager, robot, joint)

    async def is_online(self, payload: dict[str, Any], manager: PortScanner | None = None) -> bool:
        robot_payload = SO101RobotPayload(**payload)

        if manager is not None:
            ports_list = manager.robots
            ports = {p.connection_string for p in ports_list}
            if robot_payload.connection_string in ports:
                return True
            if robot_payload.serial_number:
                return any(p.serial_number == robot_payload.serial_number for p in ports_list)
            return False

        from serial.tools import list_ports

        all_ports = list_ports.comports()
        ports = {p.device for p in all_ports}
        if robot_payload.connection_string in ports:
            return True
        if robot_payload.serial_number:
            return any(p.serial_number == robot_payload.serial_number for p in all_ports)
        return False


_SO101_PROBE = SO101Probe()


def get_definitions() -> list[RobotCatalogDefinition]:
    """Return built-in SO101 robot catalog definitions."""
    return [
        RobotCatalogDefinition(
            type="SO101_Follower",
            display_name="SO101 Follower",
            role="follower",
            robot_builder=_build_so101_driver,
            robot_payload=SO101RobotPayload,
            asset=_SO101_ASSET,
            adapter_options=RobotAdapterOptions(goal_time_scale=1.0, external_effort_gain=None),
            probe=_SO101_PROBE,
        ),
        RobotCatalogDefinition(
            type="SO101_Leader",
            display_name="SO101 Leader",
            role="leader",
            robot_builder=_build_so101_driver,
            robot_payload=SO101RobotPayload,
            asset=_SO101_ASSET,
            adapter_options=RobotAdapterOptions(goal_time_scale=1.0, external_effort_gain=None),
            probe=_SO101_PROBE,
        ),
    ]
