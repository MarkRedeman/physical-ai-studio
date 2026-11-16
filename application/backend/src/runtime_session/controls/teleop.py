from __future__ import annotations

import time
from typing import TYPE_CHECKING

from robots.features import feature_names
from runtime_session.controls.base import Control
from runtime_session.controls.force_feedback import ForceFeedback
from runtime_session.store import ACTION

if TYPE_CHECKING:
    from physicalai.robot import Robot

    from runtime_session.bus import Bus
    from runtime_session.efforts import EffortChannel
    from runtime_session.store import Store
    from schemas.runtime_session import TeleopControlConfig


class TeleopControl(Control):
    """Command the follower whatever the leader measures, due by the leader's next reading.

    The leader can be anywhere when teleop starts, so for the first ``grace_s``
    each command is stamped due at the end of the grace period: the robot worker
    closes the gap over it rather than jumping. It tracks the leader the whole
    time, so the follower is on it when the grace period ends.
    """

    def __init__(
        self,
        store: Store,
        leader: Robot,
        config: TeleopControlConfig,
        bus: Bus | None = None,
        *,
        efforts: EffortChannel | None = None,
    ) -> None:
        leader_joints = tuple(feature_names(list(leader.joint_names)))
        if leader_joints != store.spec.joint_names:
            raise ValueError(
                f"The leader's joints {leader_joints} do not match the follower's {store.spec.joint_names}"
            )
        if efforts is not None and efforts.joint_names != store.spec.joint_names:
            raise ValueError("The effort channel's joints do not match the follower")
        super().__init__(store, config, name="TeleopControl", hz=config.hz, bus=bus)
        self.leader = leader
        self.grace_s = config.grace_s
        self.started_at = 0.0
        self.feedback = ForceFeedback(leader, efforts, gain=config.force_feedback_gain)

    def on_start(self) -> None:
        self.started_at = time.monotonic()
        self.feedback.start()

    def on_stop(self) -> None:
        self.feedback.clear()

    def on_tick(self) -> None:
        observation = self.leader.get_observation()
        # The follower is still aligning during the grace move; its resulting
        # efforts should not push the operator's leader around.
        if time.monotonic() >= self.started_at + self.grace_s:
            self.feedback.update()
        due = max(observation.timestamp + 1 / self.hz, self.started_at + self.grace_s)
        self.store.write(ACTION, observation.joint_positions, timestamp=due)
