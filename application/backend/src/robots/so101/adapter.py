from typing import Literal

from physicalai.robot.so101 import SO101

from robots.physicalai_adapter import PhysicalAIRobotAdapter
from schemas.robot import RobotType

RobotMode = Literal["follower", "teleoperator"]


class SO101Adapter(PhysicalAIRobotAdapter):
    name = "So101"

    def __init__(self, robot: SO101, mode: RobotMode) -> None:
        super().__init__(
            robot=robot,
            mode=mode,
            follower_type=RobotType.SO101_FOLLOWER,
            leader_type=RobotType.SO101_LEADER,
            emit_force_event_when_none=True,
            delegate_torque=True,
        )
