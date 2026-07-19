from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .schemas import SerialPortInfo


class PortScanner(Protocol):
    """Duck-type for serial port scanners (e.g. RobotConnectionManager).

    Used by RobotProbe to accept scan results without coupling to a specific
    manager implementation.
    """

    async def find_robots(self) -> None: ...

    @property
    def robots(self) -> list[SerialPortInfo]: ...


@runtime_checkable
class RobotProbe(Protocol):
    """Hardware interaction interface for a robot type.

    Each built-in or plugin-provided robot type implements this protocol
    to encapsulate device discovery, visual identification, and online
    status checking. External packages implement the protocol by defining
    a class with these three methods.
    """

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]: ...

    async def identify(
        self,
        payload: dict[str, Any],
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None: ...

    async def is_online(
        self,
        payload: dict[str, Any],
        manager: PortScanner | None = None,
    ) -> bool: ...
