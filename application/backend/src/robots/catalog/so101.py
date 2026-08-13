from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from loguru import logger
from physicalai.robot import SO101
from physicalai.robot.so101 import SO101Calibration, SO101JointCalibration
from physicalai.robot.so101.constants import TICKS_PER_REVOLUTION
from physicalai_studio_plugin import RobotAdapterOptions, RobotAsset, RobotCatalogDefinition
from pydantic import BaseModel, ConfigDict, Field, model_validator

from exceptions import RobotIdentifyError
from schemas import SerialPortInfo
from schemas.robot_type import BaseRobot

SO101Types = Literal["SO101_Follower", "SO101_Leader"]

if TYPE_CHECKING:
    from physicalai_studio_plugin import CatalogRobot, CatalogRobotFactory, PortScanner

    from schemas.robot import Robot


class SO101RobotPayload(BaseModel):
    """Connection configuration for SO-101 serial robots."""

    connection_string: str = Field(
        default="",
        description="Serial port path; leave empty to auto-discover via serial_number",
    )
    serial_number: str = Field(default="", description="USB serial number of the robot (when available)")
    calibration: dict[str, SO101JointCalibration] | None = Field(
        default=None,
        description="Per-joint calibration values (id, drive_mode, homing_offset, range_min, range_max)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "connection_string": "",
                "serial_number": "SO101-2024-001",
                "calibration": None,
            },
        },
    )

    @model_validator(mode="after")
    def validate_identifier(self) -> SO101RobotPayload:
        if self.connection_string == "" and self.serial_number == "":
            raise ValueError("Either serial_number or connection_string is required for SO101 robots")
        return self


class SO101Robot(BaseRobot):
    """SO-101 follower or leader robot using a serial connection.

    Note:
        This is a local convenience model for probe/identify helpers. Robots
        that reach a builder come from ``RobotCatalogRegistry``, which generates
        a separate model per type (``SO101_FollowerRobot``) deriving straight
        from ``BaseRobot``. Never ``isinstance``-check against this class in a
        builder — it will never match. Match on the payload type instead.
    """

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


async def _build_so101_driver(robot: CatalogRobot[SO101RobotPayload], factory: CatalogRobotFactory) -> SO101:
    # Studio generates the robot model per registered type (``SO101_LeaderRobot``),
    # so match on the payload; the model is never an ``SO101Robot`` instance.
    payload = robot.payload
    if not isinstance(payload, SO101RobotPayload):
        raise TypeError(f"Expected SO101RobotPayload, got {type(payload).__name__}")
    port_info = SerialPortInfo(
        connection_string=payload.connection_string or None,
        serial_number=payload.serial_number or None,
    )
    port = await factory.find_port(port_info)
    if port is None:
        resource_key = payload.serial_number or payload.connection_string
        raise ValueError(f"Could not resolve a serial port for {resource_key}")

    role = "follower" if robot.type == "SO101_Follower" else "leader"
    if payload.calibration is None:
        # the uncalibrated driver reports raw servo ticks (0-4095) rather
        # than the normalized range, so this is not comparable to calibrated data.
        logger.warning(
            "SO101 {} has no calibration; building in raw-ticks mode. "
            "Positions are not normalized and must not be used for policy inference.",
            robot.type,
        )
        return SO101.uncalibrated(port=port, role=role)

    return SO101(
        port=port,
        calibration=SO101Calibration(joints=payload.calibration),
        role=role,
        unit="normalized",
    )


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


# Relative wiggle (fraction of the full encoder) moved around the current
# position. Uncalibrated mode has no calibrated range, so identification only
# moves a small amount relative to where the joint already is, keeping it clear
# of the physical stops so it cannot stall and trip overload protection.
_IDENTIFY_WIGGLE = 0.08


def _identify_so101_motion(connection_string: str, joint: str) -> None:
    """Move ``joint`` a small way and back to identify the robot.

    Drives through the physicalai ``SO101`` driver in raw-ticks (uncalibrated)
    mode, which applies the gripper torque/current/overload protection
    registers on connect. Torque is disabled before disconnecting so the arm
    relaxes afterwards.
    """
    if joint not in SO101.JOINT_ORDER:
        raise ValueError(f"Unknown SO101 joint: {joint!r}")
    joint_index = SO101.JOINT_ORDER.index(joint)

    driver = SO101.uncalibrated(port=connection_string, role="follower")
    driver.torque_on_disconnect = False
    # Port open / permission failures keep the standard serial error mapping.
    driver.connect()
    try:
        observation = driver.get_observation()
        current = float(observation.joint_positions[joint_index])

        step = TICKS_PER_REVOLUTION * _IDENTIFY_WIGGLE
        lo, hi = 0.0, float(TICKS_PER_REVOLUTION - 1)
        targets = (min(max(current + step, lo), hi), min(max(current - step, lo), hi), current)

        for target in targets:
            action = observation.joint_positions.copy()
            action[joint_index] = target
            driver.send_action(action)
            time.sleep(1.0)
            # A servo that trips overload protection stops responding to sync
            # reads, so this surfaces immediately with a clear error.
            driver.get_observation()
    except ConnectionError as exc:
        raise RobotIdentifyError(
            f"Robot identify failed: {joint} stopped responding during motion. "
            "The servo may have tripped overload protection. Power-cycle the robot and try again."
        ) from exc
    finally:
        driver.set_torque(enabled=False)
        driver.disconnect()


async def identify_so101_robot_visually(
    manager: PortScanner,
    robot: Robot,
    joint: str | None = None,
) -> None:
    """Identify the robot by moving the joint a small way and back to its start.

    Uses the physicalai ``SO101`` driver in raw-ticks (uncalibrated) mode, so no
    calibration file is required. The joint is only moved a small fraction of
    its travel, never to a stop, so the STS3215 servos cannot trip overload.
    """
    if not isinstance(robot.payload, SO101RobotPayload):
        raise ValueError(f"Trying to identify unsupported robot: {robot.type}")

    if joint is None:
        joint = "gripper"

    connection_string = await find_so101_port(manager, serial_port_from_so101(robot))

    if connection_string is None:
        if robot.payload.serial_number:
            raise ValueError(f"Could not find the serial port for serial number {robot.payload.serial_number}")
        raise ValueError("Could not resolve a serial port from connection_string")

    await asyncio.to_thread(_identify_so101_motion, connection_string, joint)


class SO101Probe:
    """Probe for SO101 robots — serial port discovery + joint identification."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        await manager.find_robots()
        return manager.robots

    async def identify(
        self,
        payload: SO101RobotPayload,
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None:
        if manager is None:
            raise ValueError("PortScanner required for SO101 identification")

        now = datetime.now()
        robot = SO101Robot(
            id=UUID(int=0),
            name="",
            type="SO101_Follower",
            payload=payload,
            created_at=now,
            updated_at=now,
        )
        await identify_so101_robot_visually(manager, robot, joint)

    async def is_online(self, payload: SO101RobotPayload, manager: PortScanner | None = None) -> bool:
        serial_port = SerialPortInfo(
            connection_string=payload.connection_string or None,
            serial_number=payload.serial_number or None,
        )

        if manager is not None:
            return await find_so101_port(manager, serial_port) is not None

        from serial.tools import list_ports

        discovered = [
            SerialPortInfo(
                connection_string=port.device,
                serial_number=getattr(port, "serial_number", None) or None,
            )
            for port in list_ports.comports()
        ]
        return _resolve_serial_port(discovered, serial_port) is not None


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
