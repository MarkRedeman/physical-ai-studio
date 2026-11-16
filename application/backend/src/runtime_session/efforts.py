"""A coherent, latest-only effort sample shared with the follower process.

Unlike display telemetry, feedback must not combine values from different
readings. A spawn-safe lock protects this small vector and its timestamp.
Neither side waits for the lock: contention (or a writer killed while holding
it) makes feedback unavailable rather than blocking robot control.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from typing import TYPE_CHECKING

import numpy as np

from runtime_session.store import Sample

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class EffortChannel:
    """One follower's measured efforts, in its declared joint order.

    Pass the channel itself to a spawned worker; it is not attached by name.
    Missing or malformed sensor data invalidates the previous measurement.
    """

    def __init__(self, joint_names: tuple[str, ...]) -> None:
        self.joint_names = joint_names
        context = mp.get_context("spawn")
        self._values = context.RawArray("d", len(joint_names) + 1)
        self._lock = context.Lock()
        self._values[0] = math.nan

    def publish(self, efforts: ArrayLike | None, *, timestamp: float) -> None:
        """Publish one reading, or invalidate it if missing, nonfinite or the wrong shape."""
        try:
            values = np.asarray(efforts, dtype=np.float64) if efforts is not None else None
        except (TypeError, ValueError):
            values = None
        valid = (
            values is not None
            and values.shape == (len(self.joint_names),)
            and np.isfinite(values).all()
            and math.isfinite(timestamp)
        )
        if not self._lock.acquire(block=False):
            return
        try:
            view = np.frombuffer(self._values, dtype=np.float64)
            if valid:
                view[1:] = values
            view[0] = timestamp if valid else math.nan
        finally:
            self._lock.release()

    def invalidate(self) -> None:
        """Withdraw any previous effort measurement."""
        self.publish(None, timestamp=math.nan)

    def read(self, *, max_age_s: float, now: float | None = None) -> Sample | None:
        """Return a coherent copy only if its capture timestamp is recent and not in the future."""
        if not self._lock.acquire(block=False):
            return None
        try:
            view = np.frombuffer(self._values, dtype=np.float64)
            timestamp = float(view[0])
            age = (time.monotonic() if now is None else now) - timestamp
            if not math.isfinite(age) or not 0 <= age <= max_age_s:
                return None
            return Sample(view[1:].copy(), self.joint_names, timestamp)
        finally:
            self._lock.release()
