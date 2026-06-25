from uuid import UUID

from exceptions import ResourceNotFoundError, ResourceType
from robots.catalog.registry import RobotCatalogRegistry
from robots.physicalai_adapter import PhysicalAIRobotAdapter, PhysicalAIRobotAdapterConfig
from robots.robot_client import RobotClient
from schemas.calibration import Calibration
from schemas.robot import Robot, RobotType, SO101Robot
from services.robot_calibration_service import RobotCalibrationService
from utils.serial_robot_tools import RobotConnectionManager, normalize_serial_number


class RobotClientFactory:
    calibration_service: RobotCalibrationService
    robot_manager: RobotConnectionManager
    catalog_registry: RobotCatalogRegistry
    _active_connection_string: str | None

    def __init__(
        self,
        robot_manager: RobotConnectionManager,
        calibration_service: RobotCalibrationService,
        catalog_registry: RobotCatalogRegistry | None = None,
    ) -> None:
        self.robot_manager = robot_manager
        self.calibration_service = calibration_service
        self.catalog_registry = catalog_registry or RobotCatalogRegistry()
        self._active_connection_string = None

    def _get_robot_role(self, robot_type: RobotType):
        definition = self.catalog_registry.get_definition(robot_type)
        if definition is None:
            raise ValueError(f"Robot type is not part of the catalog: {robot_type}")
        return definition.entry.role

    async def build(self, robot: Robot) -> RobotClient:
        builder = self.catalog_registry.get_robot_builder(robot.type)
        if builder is None:
            raise ValueError(f"Unsupported robot type: {robot.type}")

        definition = self.catalog_registry.get_definition(robot.type)
        if definition is None:
            raise ValueError(f"Robot type is not part of the catalog: {robot.type}")

        self._active_connection_string = getattr(robot.payload, "connection_string", None)
        try:
            robot_driver = await builder(robot, self)
        finally:
            self._active_connection_string = None

        adapter_options = definition.adapter_options
        return PhysicalAIRobotAdapter(
            robot=robot_driver,
            robot_type=robot.type,
            robot_role=self._get_robot_role(robot.type),
            config=PhysicalAIRobotAdapterConfig(
                include_velocities=adapter_options.include_velocities,
                goal_time_scale=adapter_options.goal_time_scale,
                external_effort_gain=adapter_options.external_effort_gain,
            ),
        )

    async def find_robot_port(self, robot: SO101Robot) -> str:
        port = await self.find_port_for_robot_identifiers(
            serial_number=robot.payload.serial_number,
            connection_string=robot.payload.connection_string,
        )
        if port is None:
            resource_key = normalize_serial_number(robot.payload.serial_number) or robot.payload.connection_string
            raise ResourceNotFoundError(ResourceType.ROBOT, resource_key)
        return port

    async def get_robot_calibration(self, robot: SO101Robot) -> Calibration | None:
        if robot.active_calibration_id is None:
            return None
        return await self.calibration_service.get_calibration(robot.active_calibration_id)

    async def find_port_by_serial(self, serial_number: str) -> str | None:
        normalized_serial = normalize_serial_number(serial_number)
        if normalized_serial == "":
            return await self.find_port_by_connection_string(self._active_connection_string or "")

        for managed_robot in self.robot_manager.robots:
            if normalize_serial_number(managed_robot.serial_number) == normalized_serial:
                return managed_robot.connection_string

        await self.robot_manager.find_robots()
        for managed_robot in self.robot_manager.robots:
            if normalize_serial_number(managed_robot.serial_number) == normalized_serial:
                return managed_robot.connection_string

        return None

    async def find_port_by_connection_string(self, connection_string: str) -> str | None:
        if connection_string == "":
            return None

        for managed_robot in self.robot_manager.robots:
            if managed_robot.connection_string == connection_string:
                return managed_robot.connection_string

        await self.robot_manager.find_robots()
        for managed_robot in self.robot_manager.robots:
            if managed_robot.connection_string == connection_string:
                return managed_robot.connection_string

        return None

    async def find_port_for_robot_identifiers(self, serial_number: str, connection_string: str) -> str | None:
        normalized_serial = normalize_serial_number(serial_number)
        if normalized_serial != "":
            return await self.find_port_by_serial(normalized_serial)
        return await self.find_port_by_connection_string(connection_string)

    async def get_calibration_by_id(self, calibration_id: UUID | None) -> Calibration | None:
        if calibration_id is None:
            return None
        return await self.calibration_service.get_calibration(calibration_id)
