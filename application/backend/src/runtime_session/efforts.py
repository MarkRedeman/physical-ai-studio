"""Coherent effort snapshots shared explicitly with the spawned follower worker."""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EffortSample:
    value: np.ndarray
    timestamp: float


class EffortChannel:
    """One joint-ordered vector and timestamp, protected by a spawn-safe lock.

    Wait at most 1 ms for contention; an abandoned lock cannot stall control.
    Publishing and invalidation return False if that wait expires. Freshness
    policy belongs to the reader, not the channel.
    """

    lock_timeout_s = 0.001

    def __init__(self, joint_names: tuple[str, ...]) -> None:
        self.joint_names = joint_names
        context = mp.get_context("spawn")
        self._values = context.RawArray("d", len(joint_names) + 1)
        self._lock = context.Lock()
        self._values[0] = math.nan

    def publish(self, efforts: np.ndarray | None, *, timestamp: float) -> bool:
        """Publish or withdraw a reading; malformed driver output invalidates it and raises."""
        if efforts is not None and (
            efforts.shape != (len(self.joint_names),) or not np.isfinite(efforts).all() or not math.isfinite(timestamp)
        ):
            self.invalidate()
            raise ValueError("Efforts must be a finite joint vector with a finite capture timestamp")
        if not self._lock.acquire(timeout=self.lock_timeout_s):
            return False
        try:
            # Rebuild the view here: caching a NumPy view would pickle its data,
            # rather than preserve the shared buffer in the spawned worker.
            view = np.frombuffer(self._values, dtype=np.float64)
            if efforts is not None:
                view[1:] = efforts
            view[0] = timestamp if efforts is not None else math.nan
            return True
        finally:
            self._lock.release()

    def invalidate(self) -> bool:
        """Try to withdraw the current reading within the lock timeout."""
        return self.publish(None, timestamp=math.nan)

    def read(self) -> EffortSample | None:
        """Return a coherent copy, or None if unwritten, invalidated, or the lock wait expires."""
        if not self._lock.acquire(timeout=self.lock_timeout_s):
            return None
        try:
            view = np.frombuffer(self._values, dtype=np.float64)
            timestamp = float(view[0])
            if math.isnan(timestamp):
                return None
            return EffortSample(view[1:].copy(), timestamp)
        finally:
            self._lock.release()
