from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID


class CatalogRobotFactory(Protocol):
    """Factory protocol provided by Studio to robot builders.

    Plugin ``robot_builder`` callables receive an instance of this protocol
    so they can resolve serial ports and load calibrations at build time.
    """

    async def find_so101_port(self, robot: Any) -> str: ...

    async def find_port_by_serial(self, serial_number: str) -> str | None: ...

    async def get_calibration_by_id(
        self, calibration_id: UUID | None
    ) -> Any | None: ...
