"""Home motion tests use synthetic joint feedback; no robot connection is opened."""

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from runtime_session.home_position import HomeReturn, capability_for_so101
from runtime_session.store import Sample
from schemas.home_position import HomeCapability, HomeJointLimits, HomePose
from workers.robot_worker import RobotRecipe

NAMES = ("joint.pos", "gripper.pos")


def make_return():
    capability = HomeCapability(
        robot_id=uuid4(),
        fingerprint="a" * 64,
        unit="degrees",
        joints={
            "joint.pos": HomeJointLimits(
                minimum=-100, maximum=100, velocity=5, acceleration=10, tolerance=0.1, tracking_error=1
            )
        },
    )
    events = []
    engine = HomeReturn(capability, NAMES, events.append)
    for step in range(41):
        now = 100 + step / 100
        engine.observe(Sample(np.array([0.0, 35.0]), NAMES, now), now)
    pose = HomePose(
        robot_id=capability.robot_id,
        fingerprint=capability.fingerprint,
        unit="degrees",
        positions={"joint.pos": 3.0},
        captured_at=datetime.now(UTC),
    )
    return engine, pose, events


def test_capture_is_consistent_and_excludes_gripper():
    engine, _, _ = make_return()
    assert engine.capture(100.4).positions == {"joint.pos": 0.0}


def test_capture_rejects_unsettled_or_stale_observations():
    engine, _, _ = make_return()
    engine.observe(Sample(np.array([2.0, 35.0]), NAMES, 100.41), 100.41)
    with pytest.raises(ValueError, match="still"):
        engine.capture(100.41)
    with pytest.raises(ValueError, match="fresh"):
        engine.capture(101)


def test_return_preserves_gripper_and_obeys_velocity_acceleration_limits():
    engine, pose, events = make_return()
    operation = uuid4()
    engine.start(operation, pose, 100.4)
    measured = np.array([0.0, 35.0])
    commands = [measured.copy()]
    for step in range(1, 400):
        now = 100.4 + step / 100
        engine.observe(Sample(measured.copy(), NAMES, now), now)
        if step % 10 == 0 and engine.moving:
            engine.renew(operation, now)
        command = engine.command(now)
        if command is not None:
            measured = command
            commands.append(command.copy())
        if not engine.moving:
            break
    assert engine.motion.state == "reached"
    assert engine.motion.progress == 1
    positions = np.array(commands)
    assert np.all(positions[:, 1] == 35)
    assert np.max(np.abs(np.diff(positions[:, 0]) / 0.01)) <= 5.001
    assert np.max(np.abs(np.diff(positions[:, 0], n=2) / 0.01**2)) <= 10.01
    assert events[-1].state == "reached"


def test_already_home_does_not_start_a_move():
    engine, pose, _ = make_return()
    result = engine.start(uuid4(), pose.model_copy(update={"positions": {"joint.pos": 0}}), 100.4)
    assert result.state == "reached"
    assert engine.command(100.4) is None


def test_tracking_failure_holds_measured_pose():
    engine, pose, _ = make_return()
    engine.start(uuid4(), pose, 100.4)
    engine.observe(Sample(np.array([-2.0, 35.0]), NAMES, 100.41), 100.41)
    assert engine.command(100.41).tolist() == [-2, 35]
    assert engine.motion.state == "failed"


def test_lost_heartbeat_stops_and_late_renewal_cannot_resume():
    engine, pose, _ = make_return()
    operation = uuid4()
    engine.start(operation, pose, 100.4)
    with pytest.raises(ValueError, match="expired"):
        engine.renew(operation, 102)
    assert engine.motion.state == "failed"
    assert engine.command(102) is None


def test_delayed_tick_does_not_jump_to_target():
    engine, pose, _ = make_return()
    engine.start(uuid4(), pose, 100.4)
    engine.observe(Sample(np.array([0.0, 35.0]), NAMES, 101), 101)
    assert engine.command(101).tolist() == [0, 35]
    assert engine.motion.state == "failed"


def test_worker_watchdog_stops_without_a_browser_renewal():
    engine, pose, _ = make_return()
    engine.start(uuid4(), pose, 100.4)
    measured = np.array([0.0, 35.0])
    for step in range(1, 160):
        now = 100.4 + step / 100
        engine.observe(Sample(measured.copy(), NAMES, now), now)
        command = engine.command(now)
        if command is not None:
            measured = command
        if not engine.moving:
            break
    assert engine.motion.state == "failed"
    assert "heartbeat" in engine.motion.message


def test_cancel_and_repeated_start_do_not_restart_a_move():
    engine, pose, _ = make_return()
    operation = uuid4()
    engine.start(operation, pose, 100.4)
    assert engine.cancel(operation).state == "cancelled"
    assert engine.start(operation, pose, 100.41).state == "cancelled"
    assert not engine.moving


@pytest.mark.parametrize(
    "update",
    [
        {"robot_id": uuid4()},
        {"fingerprint": "b" * 64},
        {"unit": "normalized"},
        {"positions": {"joint.pos": 200}},
        {"positions": {"joint.pos": 0, "gripper.pos": 0}},
    ],
)
def test_incompatible_or_out_of_range_pose_is_rejected(update):
    engine, pose, _ = make_return()
    with pytest.raises(ValueError):
        engine.start(uuid4(), pose.model_copy(update=update), 100.4)
    assert not engine.moving


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "3.0"])
def test_nonfinite_or_coerced_pose_is_rejected_by_schema(value):
    _, pose, _ = make_return()
    with pytest.raises(ValidationError):
        HomePose.model_validate({**pose.model_dump(), "positions": {"joint.pos": value}})


def test_normalized_boundaries_are_not_treated_as_known_positions():
    engine, pose, _ = make_return()
    engine.capability.unit = "normalized"
    pose = pose.model_copy(update={"unit": "normalized"})
    with pytest.raises(ValueError, match="clipped"):
        engine.validate(pose.model_copy(update={"positions": {"joint.pos": 100}}))
    engine.start(uuid4(), pose, 100.4)
    with pytest.raises(ValueError, match="clipped"):
        engine.observe(Sample(np.array([100.0, 35.0]), NAMES, 100.41), 100.41)
    assert engine.motion.state == "failed"
    with pytest.raises(ValueError, match="fresh"):
        engine.hold_positions(100.41)


def test_so101_profile_requires_known_units_calibration_and_serial_identity():
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    calibration = {
        name: {"id": index + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 500, "range_max": 3500}
        for index, name in enumerate(names)
    }
    recipe = RobotRecipe(
        uuid4(),
        {"class_path": "physicalai_so101_plugin.SO101", "init_args": {"calibration": calibration}},
        tuple(f"{name}.pos" for name in names),
        ("serial:test",),
    )
    assert capability_for_so101(recipe, "") is None
    profile = capability_for_so101(recipe, "serial-number")
    assert profile is not None and profile.unit == "normalized"
    assert "gripper.pos" not in profile.joints
    recipe.driver["init_args"]["unit"] = "degrees"
    degrees = capability_for_so101(recipe, "serial-number")
    assert degrees is not None and degrees.fingerprint != profile.fingerprint
    other_environment = capability_for_so101(recipe, "serial-number", environment_id=uuid4())
    assert other_environment is not None and other_environment.fingerprint != degrees.fingerprint
    recipe.driver["init_args"]["allow_uncalibrated"] = True
    assert capability_for_so101(recipe, "serial-number") is None


def test_late_cancel_and_renew_do_not_hold_a_subsequent_control():
    from unittest.mock import Mock

    from schemas.home_position import CancelHomeReturnCommand, RenewHomeReturnCommand
    from workers.robot_worker import RobotWorker

    engine, pose, _ = make_return()
    operation = uuid4()
    engine.start(operation, pose, 100.4)
    engine.cancel(operation)
    worker = RobotWorker.__new__(RobotWorker)
    worker._home = engine
    worker._hold_home = Mock()
    worker._cancel_home(CancelHomeReturnCommand(request_id="", operation_id=operation))
    with pytest.raises(ValueError, match="no longer active"):
        worker._renew_home(RenewHomeReturnCommand(request_id="", operation_id=operation))
    worker._hold_home.assert_not_called()
