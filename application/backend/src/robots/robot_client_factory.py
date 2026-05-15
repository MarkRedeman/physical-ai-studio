from typing import Literal
from uuid import UUID

from physicalai.robot.so101 import SO101, SO101Calibration
from physicalai.robot.trossen import BimanualWidowXAI, WidowXAI

from exceptions import ResourceNotFoundError, ResourceType
from robots.physicalai_adapter import PhysicalAIRobotAdapter, PhysicalAIRobotAdapterConfig
from robots.robot_client import RobotClient
from robots.so101.bimanual_adapter import BimanualSO101Adapter
from schemas.calibration import Calibration
from schemas.robot import Robot, RobotType, SO101BimanualRobot, SO101Robot, TrossenBimanualRobot
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
                robot_driver = WidowXAI(ip=robot.payload.connection_string, role="follower")
                return PhysicalAIRobotAdapter(
                    robot=robot_driver,
                    robot_type=RobotType.TROSSEN_WIDOWXAI_FOLLOWER,
                    config=PhysicalAIRobotAdapterConfig(
                        include_velocities=True,
                        goal_time_scale=1.0,
                        external_effort_gain=0.1,
                    ),
                )
            case RobotType.TROSSEN_WIDOWXAI_LEADER:
                robot_driver = WidowXAI(ip=robot.payload.connection_string, role="leader")
                return PhysicalAIRobotAdapter(
                    robot=robot_driver,
                    robot_type=RobotType.TROSSEN_WIDOWXAI_LEADER,
                    config=PhysicalAIRobotAdapterConfig(
                        include_velocities=True,
                        goal_time_scale=1.0,
                        external_effort_gain=0.1,
                    ),
                )
            case RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER:
                return self._build_bimanual_widowxai(robot, mode="follower")
            case RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER:
                return self._build_bimanual_widowxai(robot, mode="leader")
            case RobotType.SO101_BIMANUAL_FOLLOWER:
                return await self._build_bimanual_so101(robot, mode="follower")
            case RobotType.SO101_BIMANUAL_LEADER:
                return await self._build_bimanual_so101(robot, mode="teleoperator")
            case RobotType.SO101_FOLLOWER:
                return await self._build_so101(robot)
            case RobotType.SO101_LEADER:
                return await self._build_so101(robot)
            case _:
                raise ValueError(f"Unsupported robot type: {robot.type}")

    @staticmethod
    def _build_bimanual_widowxai(
        robot: TrossenBimanualRobot, mode: Literal["follower", "leader"]
    ) -> PhysicalAIRobotAdapter:
        left_driver = WidowXAI(ip=robot.payload.connection_string_left, role=mode)
        right_driver = WidowXAI(ip=robot.payload.connection_string_right, role=mode)
        bimanual_robot = BimanualWidowXAI(left=left_driver, right=right_driver)
        robot_type = (
            RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER
            if mode == "follower"
            else RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER
        )
        return PhysicalAIRobotAdapter(
            robot=bimanual_robot,
            robot_type=robot_type,
            config=PhysicalAIRobotAdapterConfig(
                include_velocities=True,
                goal_time_scale=1.0,
                external_effort_gain=0.1,
            ),
        )

    async def _build_so101(self, robot: SO101Robot) -> PhysicalAIRobotAdapter:
        port = await self._find_robot_port(robot)
        calibration = await self._get_robot_calibration(robot)

        if calibration is None:
            raise ResourceNotFoundError(ResourceType.ROBOT_CALIBRATION, robot.payload.serial_number)
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.payload.serial_number)

        role = "follower" if robot.type == RobotType.SO101_FOLLOWER else "leader"

        so101 = SO101(port=port, calibration=self._to_so101_calibration(calibration), role=role, unit="normalized")
        return PhysicalAIRobotAdapter(
            robot=so101,
            robot_type=robot.type,
            config=PhysicalAIRobotAdapterConfig(
                include_velocities=False,
                goal_time_scale=1.0,
                external_effort_gain=None,
            ),
        )

    async def _build_bimanual_so101(
        self,
        robot: SO101BimanualRobot,
        mode: Literal["follower", "teleoperator"],
    ) -> BimanualSO101Adapter:
        left_port = await self._find_port_by_serial(robot.payload.serial_number_left)
        right_port = await self._find_port_by_serial(robot.payload.serial_number_right)

        left_calibration = await self._get_calibration_by_id(robot.payload.active_calibration_id_left)
        right_calibration = await self._get_calibration_by_id(robot.payload.active_calibration_id_right)

        if left_calibration is None:
            raise ResourceNotFoundError(ResourceType.ROBOT_CALIBRATION, robot.payload.serial_number_left)
        if right_calibration is None:
            raise ResourceNotFoundError(ResourceType.ROBOT_CALIBRATION, robot.payload.serial_number_right)
        if left_port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.payload.serial_number_left)
        if right_port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.payload.serial_number_right)

        role = "follower" if mode == "follower" else "leader"

        left_driver = SO101(port=left_port, calibration=self._to_so101_calibration(left_calibration), role=role, unit="normalized")
        right_driver = SO101(port=right_port, calibration=self._to_so101_calibration(right_calibration), role=role, unit="normalized")
        config = PhysicalAIRobotAdapterConfig(
            include_velocities=False,
            goal_time_scale=1.0,
            external_effort_gain=None,
        )
        robot_type = RobotType.SO101_FOLLOWER if role == "follower" else RobotType.SO101_LEADER

        left_adapter = PhysicalAIRobotAdapter(robot=left_driver, robot_type=robot_type, config=config,)
        right_adapter = PhysicalAIRobotAdapter(robot=right_driver, robot_type=robot_type, config=config,)

        return BimanualSO101Adapter(left=left_adapter, right=right_adapter, mode=mode)

    async def _find_robot_port(self, robot: SO101Robot) -> str:
        port = await find_robot_port(self.robot_manager, robot)
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.payload.serial_number)

        return port

    async def _get_robot_calibration(self, robot: SO101Robot) -> Calibration | None:
        if robot.active_calibration_id is None:
            return None

        return await self.calibration_service.get_calibration(robot.active_calibration_id)

    async def _find_port_by_serial(self, serial_number: str) -> str | None:
        for managed_robot in self.robot_manager.robots:
            if managed_robot.serial_number == serial_number:
                return managed_robot.connection_string
        return None

    async def _get_calibration_by_id(self, calibration_id: UUID | None) -> Calibration | None:
        if calibration_id is None:
            return None
        return await self.calibration_service.get_calibration(calibration_id)

    @staticmethod
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
