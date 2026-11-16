"""Optional bilateral effort feedback; drivers own units, direction and scaling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from physicalai.robot import Robot

    from runtime_session.efforts import EffortChannel


class EffortLeader(Protocol):
    """Optional driver capability, implemented by single and bimanual Trossen leaders.

    Input is the follower's measured efforts in matching joint order. The
    driver applies the opposing direction and ``gain`` exactly once. A zero
    vector must clear the previously commanded external efforts.
    """

    def set_external_efforts(self, efforts: np.ndarray, gain: float = 1.0) -> None: ...


class ForceFeedback:
    """Forward fresh measured efforts, clearing them when samples become unavailable.

    This is not a hardware watchdog: a blocked driver or dead backend cannot
    execute cleanup. Feedback is therefore explicitly enabled by the operator.
    """

    max_age_s = 0.1

    def __init__(self, leader: Robot, channel: EffortChannel | None, *, gain: float) -> None:
        if gain > 0 and not callable(getattr(leader, "set_external_efforts", None)):
            raise ValueError("The leader does not support force feedback")
        if gain > 0 and channel is None:
            raise ValueError("Force feedback needs the follower's effort channel")
        self._leader = cast("EffortLeader", leader)
        self._channel = channel
        self._gain = gain
        self._zeros = np.zeros(len(leader.joint_names), dtype=np.float64)
        self._engaged = False

    def start(self) -> None:
        """Clear any residual effort before teleoperation starts."""
        if self._gain > 0:
            self._engaged = True
            self.clear()

    def update(self) -> None:
        """Apply one fresh sample; failures propagate so the session stops teleoperation."""
        if self._gain == 0 or self._channel is None:
            return
        sample = self._channel.read(max_age_s=self.max_age_s)
        if sample is None:
            self.clear()
            return
        # Set before the driver call: a partial bimanual write can fail after
        # applying effort to the first arm, and still needs cleanup.
        self._engaged = True
        self._leader.set_external_efforts(sample.value, gain=self._gain)

    def clear(self) -> None:
        """Best attempted by the lifecycle even after a failed effort write; errors stay visible."""
        if self._engaged:
            self._leader.set_external_efforts(self._zeros, gain=self._gain)
            self._engaged = False
