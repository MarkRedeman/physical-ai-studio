from pathlib import Path

from physicalai.robot.so101 import SO101, SO101Calibration

from exceptions import ResourceNotFoundError, ResourceType
from schemas.calibration import Calibration
from schemas.robot import  SO101RobotPayload, RobotType, SerialPortInfo
from schemas.robot_catalog import RobotCatalogEntry

from .types import RobotAdapterOptions, RobotCatalogDefinition

_SO101_TO_URDF = {
    "shoulder_pan.pos": ["shoulder_pan"],
    "shoulder_lift.pos": ["shoulder_lift"],
    "elbow_flex.pos": ["elbow_flex"],
    "wrist_flex.pos": ["wrist_flex"],
    "wrist_roll.pos": ["wrist_roll"],
    "gripper.pos": ["gripper"],
}

async def discover_so101_devices(devices: list[SerialPortInfo]) -> list[SerialPortInfo]:
    return [d for d in devices if "so101" in d.robot_type.lower()]


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


async def _build_so101_driver(robot, factory):
    port = await factory.find_robot_port(robot)
    calibration = await factory.get_robot_calibration(robot)

    if calibration is None:
        raise ResourceNotFoundError(ResourceType.ROBOT_CALIBRATION, robot.payload.serial_number)

    role = "follower" if robot.type == RobotType.SO101_FOLLOWER else "leader"
    return SO101(port=port, calibration=_to_so101_calibration(calibration), role=role, unit="normalized")


def get_definitions() -> list[RobotCatalogDefinition]:
    urdf_relative_path = Path("SO101/so101_new_calib.urdf")
    package_root = Path("SO101")

    return [
        RobotCatalogDefinition(
            entry=RobotCatalogEntry(
                type=RobotType.SO101_FOLLOWER,
                display_name="SO101 Follower",
                role="follower",
                urdf_path=f"/api/robots/catalog/{RobotType.SO101_FOLLOWER}/urdf",
                package_map={"SO101": f"/api/robots/catalog/{RobotType.SO101_FOLLOWER}"},
                joint_map=_SO101_TO_URDF,
            ),
            urdf_relative_path=urdf_relative_path,
            package_root=package_root,
            asset_source="builtin",
            asset_root_resolver=None,
            discover_devices=discover_so101_devices,
            robot_builder=_build_so101_driver,
            payload_model=SO101RobotPayload,
            adapter_options=RobotAdapterOptions(goal_time_scale=1.0, external_effort_gain=None),
        ),
        RobotCatalogDefinition(
            entry=RobotCatalogEntry(
                type=RobotType.SO101_LEADER,
                display_name="SO101 Leader",
                role="leader",
                urdf_path=f"/api/robots/catalog/{RobotType.SO101_LEADER}/urdf",
                package_map={"SO101": f"/api/robots/catalog/{RobotType.SO101_LEADER}"},
                joint_map=_SO101_TO_URDF,
            ),
            urdf_relative_path=urdf_relative_path,
            package_root=package_root,
            asset_source="builtin",
            asset_root_resolver=None,
            discover_devices=discover_so101_devices,
            robot_builder=_build_so101_driver,
            payload_model=SO101RobotPayload,
            adapter_options=RobotAdapterOptions(goal_time_scale=1.0, external_effort_gain=None),
        ),
    ]
