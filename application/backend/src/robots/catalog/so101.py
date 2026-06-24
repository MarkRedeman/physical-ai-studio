from pathlib import Path

from physicalai.robot.so101 import SO101, SO101Calibration

from exceptions import ResourceNotFoundError, ResourceType
from robots.physicalai_adapter import PhysicalAIRobotAdapter, PhysicalAIRobotAdapterConfig
from schemas.robot import RobotType, SerialPortInfo, SO101Robot
from schemas.robot_catalog import RobotCatalogEntry
from services.robot_calibration_service import find_robot_port

from .types import RobotAdapterOptions, RobotCatalogDefinition


async def _discover_so101_devices(devices: list[SerialPortInfo]) -> list[SerialPortInfo]:
    return [d for d in devices if d.robot_type in {"SO101_Follower", "SO101_Leader"}]


async def _build_so101_driver(robot: SO101Robot, factory) -> PhysicalAIRobotAdapter:
    port = await find_robot_port(factory.robot_manager, robot)
    if port is None:
        raise ResourceNotFoundError(ResourceType.ROBOT, robot.payload.serial_number)

    calibration = await factory.get_robot_calibration(robot)
    if calibration is None:
        raise ResourceNotFoundError(ResourceType.ROBOT_CALIBRATION, robot.payload.serial_number)

    role = "follower" if robot.type == RobotType.SO101_FOLLOWER else "leader"

    so101_cal = SO101Calibration.from_dict(
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

    so101 = SO101(port=port, calibration=so101_cal, role=role, unit="normalized")
    return PhysicalAIRobotAdapter(
        robot=so101,
        robot_type=robot.type,
        config=PhysicalAIRobotAdapterConfig(
            include_velocities=False,
            goal_time_scale=1.0,
            external_effort_gain=None,
        ),
    )


_SO101_JOINT_MAP = {
    "shoulder_pan.pos": ["shoulder_pan"],
    "shoulder_lift.pos": ["shoulder_lift"],
    "elbow_flex.pos": ["elbow_flex"],
    "wrist_flex.pos": ["wrist_flex"],
    "wrist_roll.pos": ["wrist_roll"],
    "gripper.pos": ["gripper"],
}


def get_definitions() -> list[RobotCatalogDefinition]:
    urdf_relative_path = Path("so101_new_calib.urdf")
    package_root = Path("SO101")

    return [
        RobotCatalogDefinition(
            entry=RobotCatalogEntry(
                type=RobotType.SO101_FOLLOWER,
                display_name="SO101 Follower",
                role="follower",
                urdf_path=f"/api/robots/catalog/{RobotType.SO101_FOLLOWER}/urdf",
                package_map={"SO101": f"/api/robots/catalog/{RobotType.SO101_FOLLOWER}"},
                joint_map=_SO101_JOINT_MAP,
            ),
            urdf_relative_path=urdf_relative_path,
            package_root=package_root,
            asset_source="builtin",
            asset_root_resolver=None,
            discover_devices=_discover_so101_devices,
            robot_builder=_build_so101_driver,
            payload_model=None,
            adapter_options=RobotAdapterOptions(
                include_velocities=False,
                goal_time_scale=1.0,
                external_effort_gain=None,
            ),
        ),
        RobotCatalogDefinition(
            entry=RobotCatalogEntry(
                type=RobotType.SO101_LEADER,
                display_name="SO101 Leader",
                role="leader",
                urdf_path=f"/api/robots/catalog/{RobotType.SO101_LEADER}/urdf",
                package_map={"SO101": f"/api/robots/catalog/{RobotType.SO101_LEADER}"},
                joint_map=_SO101_JOINT_MAP,
            ),
            urdf_relative_path=urdf_relative_path,
            package_root=package_root,
            asset_source="builtin",
            asset_root_resolver=None,
            discover_devices=_discover_so101_devices,
            robot_builder=_build_so101_driver,
            payload_model=None,
            adapter_options=RobotAdapterOptions(
                include_velocities=False,
                goal_time_scale=1.0,
                external_effort_gain=None,
            ),
        ),
    ]
