from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from schemas.base import BaseIDModel


class SerialPortInfo(BaseModel):
    connection_string: str
    serial_number: str
    robot_type: str


class BaseRobotConfig(BaseModel):
    type: Literal["follower", "leader"]
    robot_type: str = Field(description="Robot Type")


class LeRobotConfig(BaseRobotConfig):
    type: Literal["follower", "leader"]
    robot_type: str = Field(description="Robot Type (e.g. so101)")
    id: str = Field(description="Robot calibration id")
    port: str = Field(description="Serial port of robot")
    serial_number: str = Field(description="Serial ID of device")


class NetworkIpRobotConfig(BaseRobotConfig):
    type: Literal["follower", "leader"]
    robot_type: str = Field(description="Robot Type (e.g. Trossen WidowX AI)")
    connection_string: str = Field(description="IP address of robot")


class RobotType(StrEnum):
    SO101_FOLLOWER = "SO101_Follower"
    SO101_LEADER = "SO101_Leader"
    TROSSEN_WIDOWXAI_LEADER = "Trossen_WidowXAI_Leader"
    TROSSEN_WIDOWXAI_FOLLOWER = "Trossen_WidowXAI_Follower"
    TROSSEN_BIMANUAL_WIDOWXAI_LEADER = "Trossen_Bimanual_WidowXAI_Leader"
    TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER = "Trossen_Bimanual_WidowXAI_Follower"
    REBOT_B601_DM_FOLLOWER = "ReBot_B601_DM_Follower"
    REBOT_ARM102_LEADER = "ReBot_Arm102_Leader"


# ============================================================================
# Payload Models (Configuration Only)
# ============================================================================


class SO101RobotPayload(BaseModel):
    """Connection configuration for SO-101 serial robots."""

    connection_string: str = Field(
        default="",
        description="Serial port path; leave empty to auto-discover via serial_number",
    )
    serial_number: str = Field(..., description="Unique serial number for the robot")


class TrossenSingleArmPayload(BaseModel):
    """Connection configuration for Trossen single-arm robots."""

    connection_string: str = Field(..., description="IP address of the robot")
    serial_number: str = Field(default="", description="Serial number (unused for IP robots)")


class TrossenBimanualPayload(BaseModel):
    """Connection configuration for Trossen bimanual robots."""

    connection_string_left: str = Field(..., description="IP address of the left arm")
    connection_string_right: str = Field(..., description="IP address of the right arm")
    serial_number: str = Field(default="", description="Serial number (unused for IP robots)")


class ReBotB601DMPayload(BaseModel):
    """Connection configuration for ReBot B601 DM follower."""

    connection_string: str = Field(default="", description="Serial port path; empty = auto-discover")
    serial_number: str = Field(..., description="Unique serial number for the robot")
    can_adapter: str = Field(default="damiao", description="CAN adapter type: damiao or socketcan")
    dm_serial_baud: int = Field(default=921600, description="Baud rate for DM-series serial communication")
    disable_torque_on_disconnect: bool = Field(default=True, description="Disable torque on disconnect")
    force_pos_torque_ratio: float = Field(default=0.1, description="Force-position torque ratio")


class ReBotArm102LeaderPayload(BaseModel):
    """Connection configuration for ReBot Arm 102 leader."""

    connection_string: str = Field(default="", description="Serial port path; empty = auto-discover")
    serial_number: str = Field(..., description="Unique serial number for the robot")
    baudrate: int = Field(default=1000000, description="Serial baud rate")
    unlock_on_connect: bool = Field(default=True, description="Unlock the motors on connect")
    reset_multi_turn_on_connect: bool = Field(default=True, description="Reset multi-turn angles on connect")
    zero_on_connect: bool = Field(default=False, description="Zero all joints on connect")


class CatalogRobotPayload(BaseModel):
    """Generic payload model for catalog-registered robots that accept extra fields."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="wrap")
    @classmethod
    def _passthrough(cls, data, handler):
        return data


# ============================================================================
# Concrete Robot Models
# ============================================================================


_SO101Types = Literal[RobotType.SO101_FOLLOWER, RobotType.SO101_LEADER]
_TrossenTypes = Literal[RobotType.TROSSEN_WIDOWXAI_LEADER, RobotType.TROSSEN_WIDOWXAI_FOLLOWER]
_TrossenBimanualTypes = Literal[
    RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER, RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER
]
_ReBotB601DMType = Literal[RobotType.REBOT_B601_DM_FOLLOWER]
_ReBotArm102LeaderType = Literal[RobotType.REBOT_ARM102_LEADER]


class BaseRobot(BaseIDModel):
    id: Annotated[UUID, Field(description="Unique identifier")]
    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    name: str = Field(..., description="Human-readable robot name")
    active_calibration_id: UUID | None = Field(default=None, description="The ID of the active calibration")


class SO101Robot(BaseRobot):
    """SO-101 follower or leader robot using a serial connection."""

    type: _SO101Types = Field(..., description="Type of robot configuration")
    payload: SO101RobotPayload = Field(..., description="SO-101 connection configuration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                "name": "Assembly Line Robot 1",
                "type": "SO101_Follower",
                "payload": {
                    "connection_string": "",
                    "serial_number": "SO101-2024-001",
                },
                "active_calibration_id": "b7f3d9e2-1a2b-4c3d-8e9f-0a1b2c3d4e5f",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            },
        },
    )


class TrossenSingleArmRobot(BaseRobot):
    """Trossen WidowX AI follower or leader robot using an IP connection."""

    type: _TrossenTypes = Field(..., description="Type of robot configuration")
    payload: TrossenSingleArmPayload = Field(..., description="Trossen single-arm connection configuration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                "name": "WidowX AI Robot 1",
                "type": "Trossen_WidowXAI_Follower",
                "payload": {
                    "connection_string": "192.168.1.100",
                    "serial_number": "",
                },
                "active_calibration_id": None,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            },
        },
    )


class TrossenBimanualRobot(BaseRobot):
    """Trossen Bimanual WidowX AI robot using two IP connections (left + right)."""

    type: _TrossenBimanualTypes = Field(..., description="Type of robot configuration")
    payload: TrossenBimanualPayload = Field(..., description="Trossen bimanual connection configuration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a5e2cde6-936b-4a9e-a213-08dda0afa454",
                "name": "WidowX AI Bimanual Robot 1",
                "type": "Trossen_Bimanual_WidowXAI_Follower",
                "payload": {
                    "connection_string_left": "192.168.1.100",
                    "connection_string_right": "192.168.1.101",
                    "serial_number": "",
                },
                "active_calibration_id": None,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            },
        },
    )


class ReBotB601DMRobot(BaseRobot):
    """ReBot B601 DM follower robot."""

    type: _ReBotB601DMType = Field(..., description="Type of robot configuration")
    payload: ReBotB601DMPayload = Field(..., description="ReBot B601 DM connection configuration")


class ReBotArm102LeaderRobot(BaseRobot):
    """ReBot Arm 102 leader robot."""

    type: _ReBotArm102LeaderType = Field(..., description="Type of robot configuration")
    payload: ReBotArm102LeaderPayload = Field(..., description="ReBot Arm 102 leader connection configuration")


# Discriminated union of all robot types
Robot = Annotated[
    SO101Robot | TrossenSingleArmRobot | TrossenBimanualRobot | ReBotB601DMRobot | ReBotArm102LeaderRobot,
    Field(discriminator="type"),
]

RobotAdapter: TypeAdapter[Robot] = TypeAdapter(Robot)


# ============================================================================
# RobotWithConnectionState variants
# ============================================================================

_ConnectionStatus = Literal["online", "offline", "unknown"]


class SO101RobotWithConnectionState(SO101Robot):
    connection_status: _ConnectionStatus = "unknown"


class TrossenSingleArmRobotWithConnectionState(TrossenSingleArmRobot):
    connection_status: _ConnectionStatus = "unknown"


class TrossenBimanualRobotWithConnectionState(TrossenBimanualRobot):
    connection_status: _ConnectionStatus = "unknown"


class ReBotB601DMRobotWithConnectionState(ReBotB601DMRobot):
    connection_status: _ConnectionStatus = "unknown"


class ReBotArm102LeaderRobotWithConnectionState(ReBotArm102LeaderRobot):
    connection_status: _ConnectionStatus = "unknown"


RobotWithConnectionState = Annotated[
    SO101RobotWithConnectionState
    | TrossenSingleArmRobotWithConnectionState
    | TrossenBimanualRobotWithConnectionState
    | ReBotB601DMRobotWithConnectionState
    | ReBotArm102LeaderRobotWithConnectionState,
    Field(discriminator="type"),
]

RobotWithConnectionStateAdapter: TypeAdapter[RobotWithConnectionState] = TypeAdapter(RobotWithConnectionState)
