from pydantic import BaseModel

from robots.catalog.registry import RobotCatalogRegistry
from schemas.robot import RobotType, SerialPortInfo
from schemas.robot_catalog import RobotCatalogEntry
from utils.serial_robot_tools import find_robots


class RobotCatalogService:
    _registry: RobotCatalogRegistry

    def __init__(self) -> None:
        self._registry = RobotCatalogRegistry()

    @property
    def registry(self) -> RobotCatalogRegistry:
        return self._registry

    def list_entries(self) -> list[RobotCatalogEntry]:
        return [definition.entry for definition in self._registry.list_definitions()]

    def get_entry(self, robot_type: RobotType) -> RobotCatalogEntry | None:
        definition = self._registry.get_definition(robot_type)
        return definition.entry if definition is not None else None

    async def discover_devices(self, robot_type: RobotType) -> list[SerialPortInfo]:
        devices = await find_robots()
        return await self._registry.discover_devices(robot_type, devices)

    async def is_type_online(self, robot_type: RobotType) -> bool:
        devices = await self.discover_devices(robot_type)
        return len(devices) > 0

    def validate_payload(self, robot_type: RobotType, payload: dict) -> BaseModel | dict:
        payload_model = self._registry.get_payload_model(robot_type)
        if payload_model is None:
            return payload
        return payload_model.model_validate(payload)
