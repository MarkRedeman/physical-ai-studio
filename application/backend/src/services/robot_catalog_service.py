from typing import Annotated

from fastapi import Depends

from robots.catalog.registry import RobotCatalogRegistry
from schemas.robot import RobotType, SerialPortInfo
from schemas.robot_catalog import RobotCatalogEntry
from utils.serial_robot_tools import find_robots


class RobotCatalogService:
    def __init__(self, registry: RobotCatalogRegistry | None = None) -> None:
        self._registry = registry or RobotCatalogRegistry()

    def list_entries(self) -> list[RobotCatalogEntry]:
        return [d.entry for d in self._registry.list_definitions()]

    def get_entry(self, robot_type: RobotType) -> RobotCatalogEntry | None:
        definition = self._registry.get_definition(robot_type)
        return definition.entry if definition is not None else None

    async def discover_devices(self, robot_type: RobotType) -> list[str]:
        devices: list[SerialPortInfo] = await find_robots()
        filtered = await self._registry.discover_devices(robot_type, devices)
        return [d.connection_string for d in filtered]

    async def is_type_online(self, robot_type: RobotType) -> bool:
        devices: list[SerialPortInfo] = await find_robots()
        filtered = await self._registry.discover_devices(robot_type, devices)
        return len(filtered) > 0

    def validate_payload(self, robot_type: RobotType, payload: object) -> object:
        model = self._registry.get_payload_model(robot_type)
        if model is None:
            return payload
        return model.model_validate(payload)


def get_robot_catalog_service() -> RobotCatalogService:
    return RobotCatalogService()


RobotCatalogServiceDep = Annotated[RobotCatalogService, Depends(get_robot_catalog_service)]
