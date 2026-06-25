from typing import Literal

from pydantic import BaseModel, Field

from schemas.robot import RobotType, SerialPortInfo


class RobotCatalogEntry(BaseModel):
    type: RobotType = Field(..., description="Stable backend robot type identifier")
    display_name: str = Field(..., description="Human-readable robot type label")
    role: Literal["follower", "leader"] = Field(..., description="Default robot role")
    urdf_path: str | None = Field(default=None, description="URDF URL used by the UI model loader")
    package_map: dict[str, str] = Field(default_factory=dict, description="URDF package name to URL prefix map")
    joint_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Observation joint name to URDF joint(s) mapping",
    )


class RobotCatalogOnlineResponse(BaseModel):
    type: RobotType
    online: bool


class RobotCatalogDiscoverResponse(BaseModel):
    type: RobotType
    devices: list[SerialPortInfo]
