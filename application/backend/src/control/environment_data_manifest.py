from dataclasses import dataclass, field
from typing import Any


@dataclass
class CameraManifestEntry:
    id: str
    name: str
    width: int
    height: int
    frame_data: Any  # mp.Array[c_uint8], shared with CameraWorker


@dataclass
class RobotManifestEntry:
    name: str
    type: str
    action_source: Any # mp.Value[c_int], shared with TeleoperateWorker
    features: list[str]
    state: Any  # mp.Array[c_double], shared with TeleoperateWorker
    actions: Any  # mp.Array[c_double], shared with TeleoperateWorker
    has_action_value: Any  # mp.Event, shared with TeleoperateWorker


@dataclass
class EnvironmentDataManifest:
    """Describes all data streams produced by a loaded environment."""

    robot: RobotManifestEntry
    cameras: list[CameraManifestEntry] = field(default_factory=list)
