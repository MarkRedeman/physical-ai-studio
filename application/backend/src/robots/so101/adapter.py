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
            include_velocities=False,
            convert_non_gripper_rad_to_deg=False,
            pass_goal_time=False,
            goal_time_scale=1.0,
            emit_force_event_when_none=True,
            external_effort_gain=0.1,
        )
