"""Optional bilateral effort feedback; drivers own units, direction and scaling."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from runtime_session.efforts import EffortChannel


class EffortLeader(Protocol):
    """The driver applies the opposing direction and gain; zero clears external efforts."""

    def set_external_efforts(self, efforts: np.ndarray, gain: float = 1.0) -> None: ...


class ForceFeedback:
    """An enabled feedback loop; see the architecture doc for hardware safety limitations."""

    max_age_s = 0.1

    def __init__(self, leader: EffortLeader, channel: EffortChannel, *, gain: float) -> None:
        self._leader = leader
        self._channel = channel
        self._gain = gain
        self._zeros = np.zeros(len(channel.joint_names), dtype=np.float64)
        self._engaged = True  # clear residual effort on the first lifecycle call

    def update(self) -> None:
        """Apply one fresh sample; failures propagate so the session stops teleoperation."""
        sample = self._channel.read()
        if sample is None or not 0 <= time.monotonic() - sample.timestamp <= self.max_age_s:
            self.clear()
            return
        # Set before the driver call: a partial bimanual write can fail after
        # applying effort to the first arm, and still needs cleanup.
        self._engaged = True
        self._leader.set_external_efforts(sample.value, gain=self._gain)

    def clear(self) -> None:
        """Clear previously applied efforts, propagating driver failures."""
        if self._engaged:
            self._leader.set_external_efforts(self._zeros, gain=self._gain)
            self._engaged = False
