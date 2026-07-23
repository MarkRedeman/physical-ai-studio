from __future__ import annotations

from pydantic import BaseModel


class SerialPortInfo(BaseModel):
    connection_string: str | None
    serial_number: str | None
