from fastapi import APIRouter
from frame_source import FrameSourceFactory
from loguru import logger

from schemas import CalibrationConfig, Camera, CameraProfile, Robot, SerialPortInfo
from schemas.robot import RobotType, SO101BimanualRobot, SO101Robot
from utils.calibration import get_calibrations
from utils.serial_robot_tools import find_robots, identify_so101_robot_visually
from utils.trossen_robot_tools import identify_trossen_robot_visually

router = APIRouter(prefix="/api/hardware", tags=["Hardware"])


@router.get("/cameras")
async def get_cameras() -> list[Camera]:
    """Get all cameras"""
    cameras = FrameSourceFactory.discover_devices(sources=["webcam", "realsense", "genicam", "basler"])
    logger.debug("Discovered cameras: {}", cameras)
    res = []
    sp = CameraProfile(width=640, height=480, fps=30)  # TODO: Implement proper default camera profile retrieval

    for driver, cams in cameras.items():
        for cam in cams:
            id = cam["serial_number"] if driver == "realsense" else cam["id"]
            res.append(
                Camera(
                    name=cam["name"],
                    fingerprint=id,
                    driver=driver if driver != "webcam" else "usb_camera",
                    default_stream_profile=sp,
                ),
            )
    return res


@router.get("/serial_devices")
async def get_robots() -> list[SerialPortInfo]:
    """Get all connected Robots"""
    return await find_robots()


@router.get("/calibrations")
async def get_lerobot_calibrations() -> list[CalibrationConfig]:
    """Get calibrations known to huggingface leRobot"""
    return get_calibrations()


@router.post("/identify")
async def identify_robot(robot: Robot, joint: str | None = None) -> None:
    """Visually identify the robot by moving given joint on robot"""
    if robot.type in {RobotType.SO101_LEADER, RobotType.SO101_FOLLOWER}:
        await identify_so101_robot_visually(robot, joint)

    if robot.type in {RobotType.SO101_BIMANUAL_LEADER, RobotType.SO101_BIMANUAL_FOLLOWER}:
        if not isinstance(robot, SO101BimanualRobot):
            return

        single_type = (
            RobotType.SO101_FOLLOWER if robot.type == RobotType.SO101_BIMANUAL_FOLLOWER else RobotType.SO101_LEADER
        )

        left_arm = SO101Robot.model_validate(
            {
                "id": robot.id,
                "name": f"{robot.name} Left",
                "type": single_type,
                "payload": {
                    "connection_string": robot.payload.connection_string_left,
                    "serial_number": robot.payload.serial_number_left,
                },
            }
        )
        right_arm = SO101Robot.model_validate(
            {
                "id": robot.id,
                "name": f"{robot.name} Right",
                "type": single_type,
                "payload": {
                    "connection_string": robot.payload.connection_string_right,
                    "serial_number": robot.payload.serial_number_right,
                },
            }
        )

        await identify_so101_robot_visually(left_arm, joint)
        await identify_so101_robot_visually(right_arm, joint)

    if robot.type in {RobotType.TROSSEN_WIDOWXAI_LEADER, RobotType.TROSSEN_WIDOWXAI_FOLLOWER}:
        await identify_trossen_robot_visually(robot)
