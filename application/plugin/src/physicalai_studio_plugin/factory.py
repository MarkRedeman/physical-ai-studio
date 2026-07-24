"""Factory protocol exposed to plugin robot-builder callables."""

from __future__ import annotations

from typing import Protocol


class CatalogRobotFactory(Protocol):
    """Factory protocol provided by Studio to robot builders.

    Plugin ``robot_builder`` callables receive an instance of this protocol
    so they can resolve serial ports and load calibrations at build time.
    """

    async def find_so101_port(self, robot: object) -> str:
        """Return a resolved port for a SO101-style robot descriptor."""

    async def find_port_by_serial(self, serial_number: str) -> str | None:
        """Return the connection port for a serial number, if present."""
