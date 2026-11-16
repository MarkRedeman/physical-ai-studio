from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError
from tests.workers.fake_robot import FakeRobot

from runtime_session.controls.teleop import TeleopControl
from runtime_session.efforts import EffortChannel
from runtime_session.store import ACTION, FeatureSpec, Store
from schemas.runtime_session import TeleopControlConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

JOINTS = ("shoulder_pan.pos", "gripper.pos")


@pytest.fixture
def store() -> Iterator[Store]:
    store = Store.create(FeatureSpec(JOINTS), uuid4().hex)
    yield store
    store.unlink()


def _control(store: Store, *, gain: float = 0.1, grace_s: float = 0) -> tuple[TeleopControl, Mock, EffortChannel]:
    leader = Mock(spec=["joint_names", "get_observation", "set_external_efforts"])
    leader.joint_names = ["shoulder_pan", "gripper"]
    leader.get_observation.side_effect = lambda: FakeRobot("unused", leader.joint_names, [5, 6]).get_observation()
    efforts = EffortChannel(JOINTS)
    control = TeleopControl(
        store, leader, TeleopControlConfig(force_feedback_gain=gain, grace_s=grace_s), efforts=efforts
    )
    return control, leader, efforts


def test_efforts_are_forwarded_with_gain_and_no_second_negation(store: Store) -> None:
    control, leader, efforts = _control(store)
    control.on_start()
    np.testing.assert_array_equal(leader.set_external_efforts.call_args.args[0], [0, 0])
    efforts.publish(np.array([2, -4]), timestamp=time.monotonic())
    control.on_tick()
    np.testing.assert_array_equal(leader.set_external_efforts.call_args.args[0], [2, -4])
    assert leader.set_external_efforts.call_args.kwargs == {"gain": 0.1}
    np.testing.assert_array_equal(store.read(ACTION).value, [5, 6])
    control.on_stop()
    np.testing.assert_array_equal(leader.set_external_efforts.call_args.args[0], [0, 0])


@pytest.mark.parametrize("unavailable", ["missing", "stale", "future", "invalid", "busy"])
def test_unavailable_efforts_clear_the_leader(store: Store, unavailable: str) -> None:
    control, leader, efforts = _control(store)
    control.on_start()
    efforts.publish(np.array([1, 2]), timestamp=time.monotonic())
    control.on_tick()
    if unavailable == "missing":
        efforts.invalidate()
    elif unavailable == "stale":
        efforts.publish(np.array([1, 2]), timestamp=time.monotonic() - 1)
    elif unavailable == "future":
        efforts.publish(np.array([1, 2]), timestamp=time.monotonic() + 1)
    elif unavailable == "invalid":
        with pytest.raises(ValueError, match="finite joint vector"):
            efforts.publish(np.array([np.nan, 2]), timestamp=time.monotonic())
    if unavailable == "busy":
        with efforts._lock:
            control.on_tick()
    else:
        control.on_tick()
    np.testing.assert_array_equal(leader.set_external_efforts.call_args.args[0], [0, 0])


def test_feedback_is_suppressed_during_the_grace_move(store: Store) -> None:
    control, leader, efforts = _control(store, grace_s=3)
    control.on_start()
    efforts.publish(np.array([1, 2]), timestamp=time.monotonic())
    control.on_tick()
    assert leader.set_external_efforts.call_count == 1  # only clearing at startup
    control.started_at -= 4
    control.on_tick()
    np.testing.assert_array_equal(leader.set_external_efforts.call_args.args[0], [1, 2])


def test_feedback_is_opt_in_and_does_not_require_a_capable_leader(store: Store) -> None:
    leader = FakeRobot("unused", ["shoulder_pan", "gripper"], [0, 0])
    control = TeleopControl(store, leader, TeleopControlConfig(grace_s=0))
    assert control.feedback is None
    control.on_start()
    control.on_tick()
    control.on_stop()
    with pytest.raises(ValueError, match="does not support"):
        TeleopControl(store, leader, TeleopControlConfig(force_feedback_gain=0.1))


def test_disabled_feedback_never_writes_to_a_capable_leader(store: Store) -> None:
    control, leader, efforts = _control(store, gain=0)
    efforts.publish(np.array([1, 2]), timestamp=time.monotonic())
    control.on_start()
    control.on_tick()
    control.on_stop()
    leader.set_external_efforts.assert_not_called()


def test_enabled_feedback_requires_a_channel(store: Store) -> None:
    _unused, leader, _efforts = _control(store)
    with pytest.raises(ValueError, match="matching the follower's joints"):
        TeleopControl(store, leader, TeleopControlConfig(force_feedback_gain=0.1))


def test_a_misordered_effort_channel_is_refused(store: Store) -> None:
    _unused, leader, _efforts = _control(store)
    with pytest.raises(ValueError, match="matching the follower's joints"):
        TeleopControl(
            store,
            leader,
            TeleopControlConfig(force_feedback_gain=0.1),
            efforts=EffortChannel(tuple(reversed(JOINTS))),
        )


async def test_a_partial_effort_write_failure_is_reported_and_cleared(store: Store) -> None:
    control, leader, efforts = _control(store)
    leader.set_external_efforts.side_effect = [None, OSError("second arm failed"), None]
    efforts.publish(np.array([1, 2]), timestamp=time.monotonic())
    async with control:
        await control.wait_stopped()
    assert "second arm failed" in control.failure
    assert leader.set_external_efforts.call_count == 3
    np.testing.assert_array_equal(leader.set_external_efforts.call_args.args[0], [0, 0])


async def test_leader_read_failure_still_clears_feedback(store: Store) -> None:
    control, leader, efforts = _control(store)
    # First tick succeeds; the second leader read fails.
    observation = FakeRobot("unused", leader.joint_names, [5, 6]).get_observation()
    leader.get_observation.side_effect = [observation, ConnectionError("leader lost")]
    efforts.publish(np.array([1, 2]), timestamp=time.monotonic())
    async with control:
        await control.wait_stopped()
    assert "leader lost" in control.failure
    np.testing.assert_array_equal(leader.set_external_efforts.call_args.args[0], [0, 0])


@pytest.mark.parametrize("gain", [-0.1, 1.1, np.nan, np.inf])
def test_feedback_gain_must_be_finite_and_at_most_unity(gain: float) -> None:
    with pytest.raises(ValidationError):
        TeleopControlConfig(force_feedback_gain=gain)


def test_trossen_bimanual_feedback_reaches_both_sdks_with_one_sign_and_gain(mocker) -> None:
    trossen = pytest.importorskip("physicalai_trossen_plugin")
    left = trossen.WidowXAI("192.0.2.1", role="leader")
    right = trossen.WidowXAI("192.0.2.2", role="leader")
    left_sdk, right_sdk = Mock(), Mock()
    mocker.patch.object(left, "_driver", left_sdk)
    mocker.patch.object(right, "_driver", right_sdk)
    leader = trossen.BimanualWidowXAI(left, right)
    names = tuple(f"{name}.pos" for name in leader.joint_names)
    store = Store.create(FeatureSpec(names), uuid4().hex)
    efforts = EffortChannel(names)
    observation = FakeRobot("unused", leader.joint_names, [0] * 14).get_observation()
    mocker.patch.object(leader, "get_observation", return_value=observation)
    control = TeleopControl(store, leader, TeleopControlConfig(grace_s=0, force_feedback_gain=0.1), efforts=efforts)
    try:
        control.on_start()
        values = np.arange(14, dtype=np.float64)
        efforts.publish(values, timestamp=time.monotonic())
        control.on_tick()
        np.testing.assert_allclose(list(left_sdk.set_all_external_efforts.call_args.args[0]), -0.1 * values[:7])
        np.testing.assert_allclose(list(right_sdk.set_all_external_efforts.call_args.args[0]), -0.1 * values[7:])
        control.on_stop()
        np.testing.assert_array_equal(list(left_sdk.set_all_external_efforts.call_args.args[0]), np.zeros(7))
        np.testing.assert_array_equal(list(right_sdk.set_all_external_efforts.call_args.args[0]), np.zeros(7))
    finally:
        store.unlink()
