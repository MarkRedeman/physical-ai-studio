from __future__ import annotations

import math
import multiprocessing as mp
import os

import numpy as np
import pytest

from runtime_session.efforts import EffortChannel

JOINTS = ("shoulder.pos", "gripper.pos")


def test_efforts_start_unavailable_and_round_trip_as_a_copy() -> None:
    channel = EffortChannel(JOINTS)
    assert channel.read(max_age_s=0.1, now=1) is None
    channel.publish([2, -3], timestamp=1)
    sample = channel.read(max_age_s=0.1, now=1.05)
    assert sample is not None
    assert sample.timestamp == 1
    assert sample.names == JOINTS
    np.testing.assert_array_equal(sample.value, [2, -3])
    channel.publish([4, 5], timestamp=1.05)
    np.testing.assert_array_equal(sample.value, [2, -3])


@pytest.mark.parametrize("efforts", [None, [], [1], [1, 2, 3], [[1, 2]], [math.nan, 0], [math.inf, 0], ["bad", 0]])
def test_missing_or_malformed_efforts_withdraw_the_previous_reading(efforts: object) -> None:
    channel = EffortChannel(JOINTS)
    channel.publish([1, 2], timestamp=1)
    channel.publish(efforts, timestamp=1.01)
    assert channel.read(max_age_s=0.1, now=1.02) is None


@pytest.mark.parametrize("timestamp", [0.8, 1.1, math.nan, math.inf])
def test_stale_future_or_nonfinite_timestamps_are_unavailable(timestamp: float) -> None:
    channel = EffortChannel(JOINTS)
    channel.publish([1, 2], timestamp=timestamp)
    assert channel.read(max_age_s=0.1, now=1) is None


def test_zero_efforts_are_valid_and_can_be_invalidated() -> None:
    channel = EffortChannel(JOINTS)
    channel.publish([0, 0], timestamp=1)
    assert channel.read(max_age_s=0.1, now=1) is not None
    channel.invalidate()
    assert channel.read(max_age_s=0.1, now=1) is None


def test_contention_never_blocks_or_returns_a_torn_reading() -> None:
    channel = EffortChannel(JOINTS)
    channel.publish([1, 2], timestamp=1)
    with channel._lock:
        assert channel.read(max_age_s=0.1, now=1) is None
        channel.publish([3, 4], timestamp=1)
    np.testing.assert_array_equal(channel.read(max_age_s=0.1, now=1).value, [1, 2])


def _publish(channel: EffortChannel) -> None:
    channel.publish([7, -8], timestamp=42)


def test_a_spawned_worker_publishes_to_the_same_channel() -> None:
    channel = EffortChannel(JOINTS)
    child = mp.get_context("spawn").Process(target=_publish, args=(channel,))
    child.start()
    child.join(timeout=10)
    try:
        assert child.exitcode == 0
        np.testing.assert_array_equal(channel.read(max_age_s=0.1, now=42).value, [7, -8])
    finally:
        if child.is_alive():
            child.kill()
            child.join()


def _abandon_lock(channel: EffortChannel) -> None:
    channel.publish([7, -8], timestamp=42)
    channel._lock.acquire()
    os._exit(0)


def test_a_dead_writer_cannot_leave_readers_blocked_or_return_old_efforts() -> None:
    channel = EffortChannel(JOINTS)
    child = mp.get_context("spawn").Process(target=_abandon_lock, args=(channel,))
    child.start()
    child.join(timeout=10)
    try:
        assert child.exitcode == 0
        assert channel.read(max_age_s=0.1, now=42) is None
        channel.invalidate()  # also must not block on the abandoned lock
    finally:
        if child.is_alive():
            child.kill()
            child.join()
