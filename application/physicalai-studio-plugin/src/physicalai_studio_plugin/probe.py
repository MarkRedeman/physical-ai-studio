from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from .schemas import SerialPortInfo

_PayloadT = TypeVar("_PayloadT", bound=BaseModel)


class PortScanner(Protocol):
    """Duck-type for serial port scanners (e.g. RobotConnectionManager).

    Used by RobotProbe to accept scan results without coupling to a specific
    manager implementation.
    """

    async def find_robots(self) -> None: ...

    @property
    def robots(self) -> list[SerialPortInfo]: ...


@runtime_checkable
class RobotProbe(Protocol[_PayloadT]):
    """Hardware interaction interface for a robot type.

    Each built-in or plugin-provided robot type implements this protocol
    to encapsulate device discovery, visual identification, and online
    status checking. The type parameter ``_PayloadT`` is the robot's
    payload model (a pydantic ``BaseModel`` subclass).
    """

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]: ...

    async def identify(
        self,
        payload: _PayloadT,
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None: ...

    async def is_online(
        self,
        payload: _PayloadT,
        manager: PortScanner | None = None,
    ) -> bool: ...
