from collections.abc import Mapping
from typing import NotRequired, TypedDict
from uuid import UUID

from schemas.calibration import Calibration
from schemas.environment import EnvironmentWithRelations, RobotWithTeleoperator
from schemas.project_camera import Camera
from schemas.robot import Robot


class CalibrationValueContext(TypedDict):
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


CalibrationContext = dict[str, CalibrationValueContext]


class TeleoperatorContext(TypedDict):
    type: str
    name: NotRequired[str]
    robot_type: NotRequired[str]


class EnvironmentRobotContext(TypedDict):
    name: str
    type: str
    calibration: CalibrationContext | None
    teleoperator: TeleoperatorContext


class EnvironmentCameraContext(TypedDict):
    name: str
    driver: str
    hardware_name: str | None
    width: int | None
    height: int | None
    fps: int | None
    output_type: NotRequired[str]
    depth_range_min: NotRequired[float]
    depth_range_max: NotRequired[float]


class EnvironmentContext(TypedDict):
    format: str
    version: str
    name: str
    robots: list[EnvironmentRobotContext]
    cameras: list[EnvironmentCameraContext]


def environment_context(
    environment: EnvironmentWithRelations,
    calibrations: Mapping[UUID, Calibration],
    *,
    environment_format: str = "physical_ai_studio_environment",
) -> EnvironmentContext:
    """Build the sanitized training environment payload exported with a model."""
    return {
        "format": environment_format,
        "version": "1.0",
        "name": environment.name,
        "robots": [_environment_robot_context(robot_config, calibrations) for robot_config in environment.robots],
        "cameras": [_environment_camera_context(camera) for camera in environment.cameras],
    }


def calibration_values(calibration: Calibration | None) -> CalibrationContext | None:
    """Build the LeRobot-compatible calibration values payload for export."""
    if calibration is None:
        return None

    return {
        joint_name: {
            "id": value.id,
            "drive_mode": value.drive_mode,
            "homing_offset": value.homing_offset,
            "range_min": value.range_min,
            "range_max": value.range_max,
        }
        for joint_name, value in sorted(calibration.values.items())
    }


def select_runtime_calibration(
    environment: EnvironmentWithRelations,
    calibrations: Mapping[UUID, Calibration],
) -> Calibration | None:
    """Select the first available robot calibration for runtime export."""
    for robot_config in environment.robots:
        calibration = calibrations.get(robot_config.robot.id)
        if calibration is not None:
            return calibration
    return None


def _environment_robot_context(
    robot_config: RobotWithTeleoperator,
    calibrations: Mapping[UUID, Calibration],
) -> EnvironmentRobotContext:
    robot = robot_config.robot
    payload: EnvironmentRobotContext = {
        "name": robot.name,
        "type": str(robot.type),
        "calibration": calibration_values(calibrations.get(robot.id)),
        "teleoperator": {"type": "none"},
    }

    if robot_config.tele_operator.type == "robot" and robot_config.tele_operator.robot is not None:
        payload["teleoperator"] = _teleoperator_context(robot_config.tele_operator.robot)

    return payload


def _environment_camera_context(camera: Camera) -> EnvironmentCameraContext:
    payload = camera.payload
    context: EnvironmentCameraContext = {
        "name": camera.name,
        "driver": camera.driver,
        "hardware_name": camera.hardware_name,
        "width": getattr(payload, "width", None),
        "height": getattr(payload, "height", None),
        "fps": getattr(payload, "fps", None),
    }

    if hasattr(payload, "output_type"):
        context["output_type"] = str(payload.output_type)
    if hasattr(payload, "depth_range_min"):
        context["depth_range_min"] = float(payload.depth_range_min)
    if hasattr(payload, "depth_range_max"):
        context["depth_range_max"] = float(payload.depth_range_max)

    return context


def _teleoperator_context(robot: Robot) -> TeleoperatorContext:
    return {
        "type": "robot",
        "name": robot.name,
        "robot_type": str(robot.type),
    }
