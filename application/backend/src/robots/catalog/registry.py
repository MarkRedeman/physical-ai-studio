from importlib.metadata import entry_points
from pathlib import Path

from schemas.robot import RobotType, SerialPortInfo

from . import so101, widowxai
from .types import BuildRobotCallable, RegisterPluginCallable, RobotCatalogDefinition

CATALOG_PLUGIN_ENTRYPOINT_GROUP = "physicalai.studio.catalog_plugins"


class RobotCatalogRegistry:
    _definitions: dict[RobotType, RobotCatalogDefinition]

    def __init__(self) -> None:
        self._definitions = {}
        self._register_builtin_definitions()

    def list_definitions(self) -> list[RobotCatalogDefinition]:
        return list(self._definitions.values())

    def get_definition(self, robot_type: RobotType) -> RobotCatalogDefinition | None:
        return self._definitions.get(robot_type)

    async def discover_devices(self, robot_type: RobotType, devices: list[SerialPortInfo]) -> list[SerialPortInfo]:
        definition = self._get_required_definition(robot_type)
        return await definition.discover_devices(devices)

    def get_urdf_relative_path(self, robot_type: RobotType) -> Path | None:
        definition = self.get_definition(robot_type)
        return definition.urdf_relative_path if definition is not None else None

    def get_package_root(self, robot_type: RobotType) -> Path | None:
        definition = self.get_definition(robot_type)
        return definition.package_root if definition is not None else None

    def get_asset_source(self, robot_type: RobotType) -> str | None:
        definition = self.get_definition(robot_type)
        return definition.asset_source if definition is not None else None

    def get_robot_builder(self, robot_type: RobotType) -> BuildRobotCallable | None:
        definition = self.get_definition(robot_type)
        return definition.robot_builder if definition is not None else None

    def get_payload_model(self, robot_type: RobotType):
        definition = self.get_definition(robot_type)
        return definition.payload_model if definition is not None else None

    def _register_builtin_definitions(self) -> None:
        for definition in so101.get_definitions() + widowxai.get_definitions():
            self.register(definition)

        self._load_external_plugins()

    def get_asset_root_resolver(self, robot_type: RobotType):
        definition = self.get_definition(robot_type)
        return definition.asset_root_resolver if definition is not None else None

    def register(self, definition: RobotCatalogDefinition) -> None:
        if definition.robot_type in self._definitions:
            raise ValueError(f"Duplicate robot catalog registration for type: {definition.robot_type}")
        self._definitions[definition.robot_type] = definition

    def register_many(self, definitions: list[RobotCatalogDefinition]) -> None:
        for definition in definitions:
            self.register(definition)

    def _load_external_plugins(self) -> None:
        discovered_entry_points = list(entry_points(group=CATALOG_PLUGIN_ENTRYPOINT_GROUP))

        for discovered_entry_point in discovered_entry_points:
            register_plugin = discovered_entry_point.load()
            if not callable(register_plugin):
                raise ValueError(
                    f"Catalog plugin entry point '{discovered_entry_point.name}' must load a callable "
                    "register_physicalai_studio_plugin(registry)"
                )

            plugin_callable: RegisterPluginCallable = register_plugin
            plugin_callable(self)

    def _get_required_definition(self, robot_type: RobotType) -> RobotCatalogDefinition:
        definition = self.get_definition(robot_type)
        if definition is None:
            raise ValueError(f"Robot type is not part of the catalog: {robot_type}")
        return definition
