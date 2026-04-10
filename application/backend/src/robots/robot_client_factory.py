from exceptions import ResourceNotFoundError, ResourceType
from physicalai.robot.so101 import SO101, SO101Calibration, SO101JointCalibration
from robots.physicalai_robot_client import PhysicalAIRobotClient
from robots.robot_client import RobotClient
from robots.so101.so101 import So101
from robots.widowxai.trossen_widowx_ai_follower import TrossenWidowXAIFollower
from robots.widowxai.trossen_widowx_ai_leader import TrossenWidowXAILeader
from schemas.calibration import Calibration
from schemas.robot import NetworkIpRobotConfig, Robot, RobotType
from services.robot_calibration_service import RobotCalibrationService, find_robot_port
from utils.serial_robot_tools import RobotConnectionManager


class RobotClientFactory:
    calibration_service: RobotCalibrationService
    robot_manager: RobotConnectionManager

    def __init__(
        self,
        robot_manager: RobotConnectionManager,
        calibration_service: RobotCalibrationService,
    ) -> None:
        self.robot_manager = robot_manager
        self.calibration_service = calibration_service

    async def build(self, robot: Robot) -> RobotClient:
        match robot.type:
            case RobotType.TROSSEN_WIDOWXAI_FOLLOWER:
                config = NetworkIpRobotConfig(
                    type="follower",
                    robot_type=RobotType.TROSSEN_WIDOWXAI_FOLLOWER,
                    connection_string=robot.connection_string,
                )
                return TrossenWidowXAIFollower(config=config)
            case RobotType.TROSSEN_WIDOWXAI_LEADER:
                config = NetworkIpRobotConfig(
                    type="leader",
                    robot_type=RobotType.TROSSEN_WIDOWXAI_LEADER,
                    connection_string=robot.connection_string,
                )
                return TrossenWidowXAILeader(config=config)
            case RobotType.SO101_FOLLOWER:
                return await self._build_physicalai_so101(robot)
            case RobotType.SO101_LEADER:
                return await self._build_physicalai_so101(robot)
            case _:
                raise ValueError(f"Unsupported robot type: {robot.type}")

    async def _build_so101(self, robot: Robot) -> So101:
        port = await self._find_robot_port(robot)
        calibration = await self._get_robot_calibration(robot)

        if calibration is None:
            raise ResourceNotFoundError(ResourceType.ROBOT_CALIBRATION, robot.serial_number)
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.serial_number)
        mode = "follower" if robot.type == RobotType.SO101_FOLLOWER else "teleoperator"
        return So101(port=port, id=robot.name.lower(), mode=mode, calibration=calibration)

    async def _build_physicalai_so101(self, robot: Robot) -> PhysicalAIRobotClient:
        port = await self._find_robot_port(robot)
        calibration = await self._get_robot_calibration(robot)

        if calibration is None:
            raise ResourceNotFoundError(ResourceType.ROBOT_CALIBRATION, robot.serial_number)
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.serial_number)

        so101_calibration = self._convert_to_so101_calibration(calibration)
        role = "follower" if robot.type == RobotType.SO101_FOLLOWER else "leader"
        so101 = SO101(port=port, calibration=so101_calibration, role=role)

        return PhysicalAIRobotClient(
            robot=so101,
            robot_type_value=robot.type,
            joint_names=SO101.JOINT_ORDER,
            name=robot.name.lower(),
        )

    async def _find_robot_port(self, robot: Robot) -> str:
        port = await find_robot_port(self.robot_manager, robot)
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.serial_number)

        return port

    async def _get_robot_calibration(self, robot: Robot) -> Calibration | None:
        if robot.active_calibration_id is None:
            return None

        return await self.calibration_service.get_calibration(robot.active_calibration_id)

    @staticmethod
    def _convert_to_so101_calibration(calibration: Calibration) -> SO101Calibration:
        """Convert a backend Calibration schema to a physicalai SO101Calibration.

        Two calibration differences must be reconciled:

        1. **Homing offset in firmware**: LeRobot writes ``homing_offset``
           into each motor's EPROM so the firmware subtracts it from every
           ``Present_Position`` read. The physicalai driver also subtracts
           ``homing_offset`` in software, so we must not pass the original
           value or it would be applied twice.

        2. **Zero-point convention**: LeRobot normalizes positions to
           ``[-100, +100]`` where 0 is the midpoint of ``[range_min, range_max]``.
           The physicalai driver's zero is at ``ticks == homing_offset``.
           To preserve the same zero-point, we set ``homing_offset`` to the
           range midpoint so that the midpoint maps to 0 radians/degrees.

        The ``range_min``/``range_max`` values are already in the
        firmware-offset-corrected tick space and remain valid for clamping.
        """
        joints = {
            name: SO101JointCalibration(
                id=cal_value.id,
                drive_mode=cal_value.drive_mode,
                homing_offset=round((cal_value.range_min + cal_value.range_max) / 2),
                range_min=cal_value.range_min,
                range_max=cal_value.range_max,
            )
            for name, cal_value in calibration.values.items()
        }
        return SO101Calibration(joints=joints)
