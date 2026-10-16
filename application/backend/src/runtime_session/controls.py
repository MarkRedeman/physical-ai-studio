"""What drives the follower: one control at a time, writing ``action``.

A control is a thread in the session's process, ticking at its own rate. It
borrows what the session owns (the store, the leader) and never connects or
disconnects anything, so switching controls is cheap and leaves every device
as it was. The robot worker follows whatever ``action`` holds, at its own rate.

See the "Controls" section of docs/feature_plans/runtime-inhousing-plan.md.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from robots.features import feature_names
from runtime_session.store import ACTION, STATE
from schemas.runtime_session import MoveControlConfig, TeleopControlConfig
from workers.base import ThreadWorker
from workers.rate import Rate

if TYPE_CHECKING:
    from physicalai.robot import Robot

    from runtime_session.store import Store
    from schemas.runtime_session import ControlConfig


class Control(ThreadWorker[None]):
    """Calls :meth:`tick` at ``config.hz`` until stopped. An exception in ``tick`` ends the control."""

    def __init__(self, store: Store, config: ControlConfig, *, name: str) -> None:
        super().__init__(name=name)
        self.store = store
        self.config = config
        """What was asked for; the session reports it in its state."""

    async def run_loop(self, context: None) -> None:  # noqa: ARG002
        rate = Rate(self.config.hz, name=self.name)
        while not self.should_stop():
            self.tick()
            await rate.sleep()

    @abstractmethod
    def tick(self) -> None:
        """Decide and write ``action``."""


class TeleopControl(Control):
    """Command the follower whatever the leader measures, stamped with the leader's capture time."""

    def __init__(self, store: Store, leader: Robot, config: TeleopControlConfig) -> None:
        leader_joints = tuple(feature_names(list(leader.joint_names), include_velocities=False))
        if leader_joints != store.spec.joint_names:
            raise ValueError(
                f"The leader's joints {leader_joints} do not match the follower's {store.spec.joint_names}"
            )
        super().__init__(store, config, name="TeleopControl")
        self.leader = leader

    def tick(self) -> None:
        observation = self.leader.get_observation()
        self.store.write(ACTION, observation.joint_positions, timestamp=observation.timestamp)


class MoveControl(Control):
    """Interpolate from where the follower is to a target over ``duration_s``, then hold the target."""

    def __init__(self, store: Store, config: MoveControlConfig) -> None:
        target = config.position
        unknown = set(target) - set(store.spec.joint_names)
        if unknown:
            raise ValueError(f"Unknown joints {sorted(unknown)}; the follower has {list(store.spec.joint_names)}")
        state = store.read(STATE)
        if state is None:
            raise ValueError("The follower has not been measured yet")
        super().__init__(store, config, name="MoveControl")
        self.start_position = state.value
        self.target = np.array([target.get(name, start) for name, start in zip(state.names, state.value, strict=True)])
        self.duration_s = config.duration_s
        self.started_at = time.monotonic()

    def tick(self) -> None:
        now = time.monotonic()
        fraction = min((now - self.started_at) / self.duration_s, 1.0)
        self.store.write(ACTION, self.start_position + (self.target - self.start_position) * fraction, timestamp=now)


def build_control(config: ControlConfig, store: Store, leader: Robot | None) -> Control:
    """Raises ``ValueError`` when the environment cannot run ``config``."""
    if isinstance(config, TeleopControlConfig):
        if leader is None:
            raise ValueError("Teleoperation needs a leader arm; this environment has none")
        return TeleopControl(store, leader, config)
    return MoveControl(store, config)
