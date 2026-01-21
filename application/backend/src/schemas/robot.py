from abc import ABC
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from schemas.base import BaseIDModel


class RobotPortInfo(BaseModel):
    port: str
    serial_id: str
    robot_type: str


class RobotConfig(BaseModel):
    id: str = Field(description="Robot calibration id")
    type: Literal["follower", "leader"]
    port: str = Field(description="Serial port of robot")
    serial_id: str = Field(description="Serial ID of device")
    robot_type: str = Field(description="Robot Type (e.g. so101)")


class RobotType(StrEnum):
    SO101_FOLLOWER = "SO101_Follower"
    SO101_LEADER = "SO101_Leader"


SupportedRobotDriver = Literal[
    "feetech",
    "virtual", # virtual state handled by Geti Action
    # "gym", 
    "websocket",
    "zmq",
]


class FeetechRobotPayload(BaseModel):
    """Configuration for Robot."""

    serial_id: str = Field(..., description="Unique serial identifier for the robot")

class WebsocketRobotPayload(BaseModel):
    """Configuration for Robot."""

    stream_url: str = Field(..., description="WebSocket stream URL")


class BaseRobot(BaseIDModel, ABC):
    driver: SupportedRobotDriver = "feetech"

    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    name: str = Field(..., description="Human-readable robot name")
    serial_id: str = Field(..., description="Unique serial identifier for the robot")
    type: RobotType = Field(..., description="Type of robot configuration")
    active_calibration_id: UUID | None = Field(default=None, description="The ID of the active calibration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                "name": "Assembly Line Robot 1",
                "serial_id": "SO101-2024-001",
                "type": "SO101_Leader",
                "active_calibration_id": "b7f3d9e2-1a2b-4c3d-8e9f-0a1b2c3d4e5f",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        }
    )


class FeetechRobot(BaseRobot):
    driver: Literal["feetech"] = "feetech"  # type: ignore[assignment]
    payload: FeetechRobotPayload = FeetechRobotPayload(
        serial_id = ""
    )

class WebSocketRobot(BaseRobot):
    driver: Literal["websocket"] = "websocket"  # type: ignore[assignment]
    payload: WebsocketRobotPayload 


class VirtualRobot(BaseRobot):
    driver: Literal["virtual"] = "virtual"  # type: ignore[assignment]
    #payload: WebsocketRobotPayload 

class ZMQRobot(BaseRobot):
    driver: Literal["zqm"] = "zmq"  # type: ignore[assignment]

Robot = Annotated[
    FeetechRobot | WebSocketRobot | VirtualRobot | ZMQRobot,
    Field(discriminator="driver"),
]

RobotAdapter: TypeAdapter[Robot] = TypeAdapter(Robot)
