from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import BaseModel, Field

from physicalai_studio_plugin import (
    CatalogRobotFactory,
    PortScanner,
    RobotAdapterOptions,
    RobotAsset,
    RobotCatalogDefinition,
    RobotProbe,
    SerialPortInfo,
)


class TestPayload(BaseModel):
    serial_number: str = Field(...)
    connection_string: str = ""


class TestProbe:
    """Structurally implements RobotProbe[TestPayload]."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        return []

    async def identify(
        self,
        payload: TestPayload,
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None:
        self._last_payload = payload

    async def is_online(
        self,
        payload: TestPayload,
        manager: PortScanner | None = None,
    ) -> bool:
        return payload.serial_number != ""


class TestTypedProtocol(unittest.TestCase):
    """Verify that RobotProbe properly types its payload parameter."""

    def test_probe_is_runtime_checkable(self) -> None:
        probe = TestProbe()
        self.assertIsInstance(probe, RobotProbe)

    def test_typed_payload_reaches_identify(self) -> None:
        probe = TestProbe()
        payload = TestPayload(serial_number="SN-001")

        import asyncio

        asyncio.run(probe.identify(payload, None))
        self.assertIs(probe._last_payload, payload)

    def test_is_online_typed_payload(self) -> None:
        probe = TestProbe()
        payload = TestPayload(serial_number="SN-001")

        import asyncio

        result = asyncio.run(probe.is_online(payload))
        self.assertTrue(result)

    def test_is_online_empty_payload(self) -> None:
        probe = TestProbe()
        payload = TestPayload(serial_number="")

        import asyncio

        result = asyncio.run(probe.is_online(payload))
        self.assertFalse(result)


class TestGenericRobotCatalogDefinition(unittest.TestCase):
    """Verify RobotCatalogDefinition generic wiring."""

    def test_definition_creation(self) -> None:
        asset = RobotAsset(
            urdf_relative_path=Path("test/model.urdf"),
            packages={"test": Path("test")},
            joint_map={"gripper.pos": ["gripper"]},
        )
        probe = TestProbe()
        definition = RobotCatalogDefinition[TestPayload](
            type="Test_Follower",
            display_name="Test Follower",
            role="follower",
            robot_payload=TestPayload,
            asset=asset,
            adapter_options=RobotAdapterOptions(include_velocities=True),
            probe=probe,
        )

        self.assertEqual(definition.type, "Test_Follower")
        self.assertEqual(definition.robot_payload, TestPayload)
        self.assertIs(definition.probe, probe)
        self.assertEqual(definition.robot_type, "Test_Follower")

    def test_generic_payload_linked_to_probe(self) -> None:
        """robot_payload type is the same type used by the probe."""
        asset = RobotAsset(
            urdf_relative_path=Path("test/model.urdf"),
            packages={"test": Path("test")},
            joint_map={"gripper.pos": ["gripper"]},
        )
        probe = TestProbe()
        definition = RobotCatalogDefinition[TestPayload](
            type="Test_Follower",
            display_name="Test Follower",
            role="follower",
            robot_payload=TestPayload,
            asset=asset,
            probe=probe,
        )

        import asyncio

        payload_instance = TestPayload(serial_number="SN-002")
        asyncio.run(definition.probe.identify(payload_instance, None))
        self.assertIs(probe._last_payload, payload_instance)

    def test_multiple_definitions(self) -> None:
        """list[RobotCatalogDefinition] holds mixed types."""
        definitions: list[RobotCatalogDefinition] = [
            RobotCatalogDefinition[TestPayload](
                type="RobotA",
                display_name="Robot A",
                role="follower",
                robot_payload=TestPayload,
                probe=TestProbe(),
            ),
            RobotCatalogDefinition[TestPayload](
                type="RobotB",
                display_name="Robot B",
                role="leader",
                robot_payload=TestPayload,
                probe=TestProbe(),
            ),
        ]
        self.assertEqual(len(definitions), 2)
        self.assertEqual(definitions[0].type, "RobotA")
        self.assertEqual(definitions[1].role, "leader")


class TestValidateProbePayload(unittest.TestCase):
    """Simulate the API-level _validate_probe_payload logic."""

    def test_valid_payload_passes_validation(self) -> None:
        from pydantic import BaseModel

        asset = RobotAsset(
            urdf_relative_path=Path("test/model.urdf"),
            packages={"test": Path("test")},
            joint_map={"gripper.pos": ["gripper"]},
        )
        definition = RobotCatalogDefinition[TestPayload](
            type="Test_Follower",
            display_name="Test Follower",
            role="follower",
            robot_payload=TestPayload,
            asset=asset,
        )

        raw = {"serial_number": "SN-003", "connection_string": "/dev/ttyUSB0"}
        validated = definition.robot_payload.model_validate(raw)
        self.assertIsInstance(validated, BaseModel)
        self.assertEqual(validated.serial_number, "SN-003")

    def test_invalid_payload_raises(self) -> None:
        definition = RobotCatalogDefinition[TestPayload](
            type="Test_Follower",
            display_name="Test Follower",
            role="follower",
            robot_payload=TestPayload,
        )

        raw = {"connection_string": "/dev/ttyUSB0"}  # missing serial_number
        with self.assertRaises(Exception):
            definition.robot_payload.model_validate(raw)

    def test_no_payload_model_returns_raw_dict(self) -> None:
        definition = RobotCatalogDefinition[TestPayload](
            type="NoPayload",
            display_name="No Payload",
            role="follower",
            robot_payload=None,
        )

        raw = {"some": "data"}
        result = raw if definition.robot_payload is None else definition.robot_payload.model_validate(raw)
        self.assertEqual(result, raw)
