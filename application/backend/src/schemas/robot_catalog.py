from pydantic import BaseModel, Field

from schemas.robot import RobotType


class RobotCatalogEntry(BaseModel):
    type: RobotType = Field(..., description="Robot type discriminator")
    display_name: str = Field(..., description="Human-readable name for UI")
    role: str = Field(..., description="Robot role: follower or leader")
    urdf_path: str | None = Field(None, description="URL path to the URDF model")
    package_map: dict[str, str] = Field(default_factory=dict, description="URDF package URL mappings")
    joint_map: dict[str, list[str]] = Field(default_factory=dict, description="Observation-to-URDF joint mapping")
    asset_source: str = Field("builtin", description="Where robot assets are sourced: builtin or plugin")


class RobotCatalogOnlineResponse(BaseModel):
    online: bool


class RobotCatalogDiscoverResponse(BaseModel):
    ports: list[str]
