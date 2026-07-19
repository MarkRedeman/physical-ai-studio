from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from schemas.base import BaseIDModel

RobotType = str


class SerialPortInfo(BaseModel):
    connection_string: str | None
    serial_number: str | None


class BaseRobot(BaseIDModel):
    id: Annotated[UUID, Field(description="Unique identifier")]
    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    name: str = Field(..., description="Human-readable robot name")
    active_calibration_id: UUID | None = Field(default=None, description="The ID of the active calibration")
