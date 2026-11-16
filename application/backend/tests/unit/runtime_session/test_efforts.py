from __future__ import annotations

import math
import multiprocessing as mp
import os
import threading

import numpy as np
import pytest

from runtime_session.efforts import EffortChannel

JOINTS = ("shoulder.pos", "gripper.pos")


def test_efforts_start_unavailable_and_round_trip_as_a_copy() -> None:
    channel = EffortChannel(JOINTS)
    assert channel.read() is None
    assert channel.publish(np.array([2, -3]), timestamp=1)
    sample = channel.read()
    assert sample is not None
    assert sample.timestamp == 1
    np.testing.assert_array_equal(sample.value, [2, -3])
    channel.publish(np.array([4, 5]), timestamp=1.05)
    np.testing.assert_array_equal(sample.value, [2, -3])


@pytest.mark.parametrize("efforts", [[], [1], [1, 2, 3], [[1, 2]], [math.nan, 0], [math.inf, 0]])
def test_malformed_efforts_are_reported_and_withdraw_the_previous_reading(efforts: list) -> None:
    channel = EffortChannel(JOINTS)
    channel.publish(np.array([1, 2]), timestamp=1)
    with pytest.raises(ValueError, match="finite joint vector"):
        channel.publish(np.array(efforts), timestamp=1.01)
    assert channel.read() is None


@pytest.mark.parametrize("timestamp", [math.nan, math.inf])
def test_nonfinite_timestamps_are_reported(timestamp: float) -> None:
    channel = EffortChannel(JOINTS)
    with pytest.raises(ValueError, match="finite capture timestamp"):
        channel.publish(np.array([1, 2]), timestamp=timestamp)
    assert channel.read() is None


def test_zero_efforts_are_valid_and_can_be_invalidated() -> None:
    channel = EffortChannel(JOINTS)
    channel.publish(np.array([0, 0]), timestamp=1)
    assert channel.read() is not None
    assert channel.invalidate()
    assert channel.read() is None


def test_missing_efforts_withdraw_the_previous_reading() -> None:
    channel = EffortChannel(JOINTS)
    channel.publish(np.array([1, 2]), timestamp=1)
    assert channel.publish(None, timestamp=2)
    assert channel.read() is None


def test_lock_timeouts_are_reported_without_returning_a_torn_reading() -> None:
    channel = EffortChannel(JOINTS)
    channel.publish(np.array([1, 2]), timestamp=1)
    with channel._lock:
        assert channel.read() is None
        assert not channel.publish(np.array([3, 4]), timestamp=1)
        assert not channel.invalidate()
    np.testing.assert_array_equal(channel.read().value, [1, 2])


@pytest.mark.parametrize("operation", ["read", "invalidate"])
def test_short_contention_waits_for_a_coherent_snapshot(operation: str) -> None:
    channel = EffortChannel(JOINTS)
    channel.publish(np.array([1, 2]), timestamp=1)
    channel.lock_timeout_s = 1  # allow scheduling slack in the test
    channel._lock.acquire()
    release = threading.Timer(0.01, channel._lock.release)
    release.start()
    try:
        result = getattr(channel, operation)()
        if operation == "read":
            np.testing.assert_array_equal(result.value, [1, 2])
        else:
            assert result
            assert channel.read() is None
    finally:
        release.join()


def _publish(channel: EffortChannel) -> None:
    channel.publish(np.array([7, -8]), timestamp=42)


def test_a_spawned_worker_publishes_to_the_same_channel() -> None:
    channel = EffortChannel(JOINTS)
    child = mp.get_context("spawn").Process(target=_publish, args=(channel,))
    child.start()
    child.join(timeout=10)
    try:
        assert child.exitcode == 0
        np.testing.assert_array_equal(channel.read().value, [7, -8])
    finally:
        if child.is_alive():
            child.kill()
            child.join()


def _abandon_lock(channel: EffortChannel) -> None:
    channel.publish(np.array([7, -8]), timestamp=42)
    channel._lock.acquire()
    os._exit(0)


def test_a_dead_writer_cannot_leave_readers_blocked_or_return_old_efforts() -> None:
    channel = EffortChannel(JOINTS)
    child = mp.get_context("spawn").Process(target=_abandon_lock, args=(channel,))
    child.start()
    child.join(timeout=10)
    try:
        assert child.exitcode == 0
        assert channel.read() is None
        assert not channel.invalidate()
    finally:
        if child.is_alive():
            child.kill()
            child.join()
